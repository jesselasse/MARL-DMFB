import torch
import os
from network.base_net import *
from network.vdn_net import VDNNet


class VDN:
    def __init__(self, nargs, model_dir, targs=None, cuda=True):
        self.n_actions = nargs.n_actions
        self.cuda=cuda
        # self.n_agents = args.n_agents

        # 根据参数决定RNN的输入维度
        # if nargs.last_action:
        #     input_shape += self.n_actions
        # if args.reuse_network:
        #     input_shape += self.n_agents

        # 神经网络
        if nargs.net == 'rnn':
            print('not support rnn now')
            # self.eval_rnn = RNN(input_shape, nargs)  # 每个agent选动作的网络
            # self.target_rnn = RNN(input_shape, nargs)
        elif nargs.net == 'crnn':

            self.eval_rnn = CRNN(nargs)  # 每个agent选动作的网络
            self.target_rnn = CRNN(nargs)
        if nargs.net == 'FC4':
            input_shape = nargs.obs_shape
            if nargs.last_action:
               input_shape += self.n_actions
            self.eval_rnn = FC4(input_shape, self.n_actions, hd=nargs.rnn_hidden_dim)
            self.target_rnn = FC4(input_shape, self.n_actions, hd=nargs.rnn_hidden_dim)
        elif nargs.net == 'CRatt':

            self.eval_rnn = CRnnattFC1(nargs)  # 每个agent选动作的网络
            self.target_rnn = CRnnattFC1(nargs)
        self.eval_vdn_net = VDNNet()  # 把agentsQ值加起来的网络
        self.target_vdn_net = VDNNet()
        self.nargs = nargs
        if self.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_vdn_net.cuda()
            self.target_vdn_net.cuda()

        # self.model_dir = args.model_dir + '/' + args.alg + '/' + '{}by{}-{}d{}b'.format(
        #     self.args.width, self.args.length, self.args.drop_num, self.args.block_num)+'/'
        # self.model_dir = args.model_dir + '/' + args.alg + '/fov{}/{}d{}b'.format(args.fov,
        #      self.args.drop_num, self.args.block_num)+'/'
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # 如果存在模型则加载模型
        if self.nargs.load_model:
            self.load_model_old(nargs.load_model_name)

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg VDN')

        if targs is None:
            return
        self.args=targs
        self.eval_parameters = list(
            self.eval_vdn_net.parameters()) + list(self.eval_rnn.parameters())
        if targs.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(
                self.eval_parameters, lr=targs.lr)
        elif targs.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.eval_parameters, lr=targs.lr)
        elif targs.optimizer == 'ADAM':
            self.optimizer = torch.optim.Adam(
                self.eval_parameters, lr=targs.lr, betas=(0.9, 0.99))
        elif targs.optimizer == 'ASGD':
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=targs.lr)


    # train_step表示是第几次学习，用来控制更新target_net网络的参数
    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['u'].shape[0]
        n_agents=batch['u'].shape[2]
        self.init_hidden(episode_num, n_agents)
        u, r, avail_u, avail_u_next, terminated = batch['u'], batch['r'], batch['avail_u'], \
            batch['avail_u_next'], batch['terminated']  # avail_u_next干嘛用的？
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        if self.cuda:
            u = u.cuda().long()
            r = r.cuda()
            mask = mask.cuda()
            terminated = terminated.cuda().float()
        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents，n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)

        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        # if max_episode_len<40:
        #      print(max_episode_len)
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # 得到target_q
        # q_targets[avail_u_next == 0.0] = - 9999999 #不合法动作不能选
        q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_vdn_net(q_evals)
        q_total_target = self.target_vdn_net(q_targets)

        targets = r + self.args.gamma * q_total_target * (1 - terminated) # ?

        td_error = targets.detach() - q_total_eval
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # loss = masked_td_error.pow(2).mean()
        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        # print('Loss is ', loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

    def _get_inputs(self, batch, transition_idx):
        # 给obs加上last_action、agent_id作为网络的输入
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        episode_num = batch['o'].shape[0]
        inputs= []
        inputs.append(batch['o'][:, transition_idx])
        # 给obs添加上一个动作、agent编号
        if self.nargs.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(batch['u_onehot'][:, transition_idx]))
            else:
                inputs.append(batch['u_onehot'][:, transition_idx - 1])

        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成episode_num*n_agents条数据
        # 因为这里所有agent共享一个神经网络，（每条数据中带上了自己的编号，所以还是自己的数据）不要编号了
        inputs = torch.cat(inputs, dim=2).flatten(end_dim=1) #把Input agent和episode拼起来
        # inputs shape: (N*e,obs+Acion_N)
        return inputs

    def get_q_values_p(self, batch, max_episode_len, n_agents):
        episode_num = batch['u'].shape[0] # batch_size
        q_evals, q_targets = [], []
        inputs=self._get_inputs(
                batch, 0)  # 给obs加last_action
        if self.cuda:
            inputs = inputs.cuda()
        for transition_idx in range(max_episode_len):
            inputs_next = self._get_inputs(
                batch, transition_idx+1)  # 给obs加last_action
            if self.cuda:
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(inputs,
                                                     self.eval_hidden)  # 得到的q_eval维度为(episode_num*n_agents, n_actions)
            q_target, self.target_hidden = self.target_rnn(
                inputs_next, self.target_hidden)

            # 把q_eval维度重新变回(episode_num, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, n_agents, -1)
            q_target = q_target.view(episode_num, n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
            inputs = inputs_next
        # 得的q_evals和q_targets是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def get_q_values(self, batch, max_episode_len):
        q_evals, q_targets = [], []
        inputs = batch['o'][:, 0]
        if self.cuda:
            inputs = inputs.cuda()
        for transition_idx in range(max_episode_len):
            inputs_next = batch['o'][:, transition_idx+1]
            if self.cuda:
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(inputs,
                                                     self.eval_hidden)  # 得到的q_eval维度为(episode_num*n_agents, n_actions)
            q_target, self.target_hidden = self.target_rnn(
                inputs_next, self.target_hidden)
            inputs = inputs_next
            q_evals.append(q_eval)
            q_targets.append(q_target)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num, n_agents):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros(
            (episode_num, n_agents, self.nargs.rnn_hidden_dim))
        self.target_hidden = torch.zeros(
            (episode_num, n_agents, self.nargs.rnn_hidden_dim))

    def save_model(self, i, train_step=None):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        i = str(i)
        if train_step is None:
            torch.save(self.eval_vdn_net.state_dict(),
                       self.model_dir + i + '_' + 'vdn_net_params.pkl')
            torch.save(self.eval_rnn.state_dict(),
                       self.model_dir + i + '_' + 'rnn_net_params.pkl')
        else:
            num = str(train_step)
            torch.save(self.eval_vdn_net.state_dict(),
                       self.model_dir + i + '_'+num + '_vdn_net_params.pkl')
            torch.save(self.eval_rnn.state_dict(), self.model_dir + i + '_' + num + '_rnn_net_params.pkl')


    def load_model_old(self, load_model_name):
        path_rnn = self.model_dir + load_model_name + 'rnn_net_params.pkl'
        path_vdn = self.model_dir + load_model_name + 'vdn_net_params.pkl'
        if os.path.exists(path_rnn):
            map_location = 'cuda:0' if self.cuda else 'cpu'
            self.eval_rnn.load_state_dict(torch.load(
                path_rnn, map_location=map_location))
            self.eval_vdn_net.load_state_dict(
                torch.load(path_vdn, map_location=map_location))
            print('Successfully load the model: {} and {}'.format(
                path_rnn, path_vdn))
        else:
            raise Exception("No model!")

    def save_model_new(self,save_name):
        torch.save({'vdn_state_dict': self.eval_vdn_net.state_dict(),
                       'rnn_state_dict': self.eval_rnn.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                                       self.model_dir + save_name + '_model.pth')

    def load_model_new(self,load_name):
        if os.path.exists(self.model_dir + load_name + '_model.pth'):
            map_location = 'cuda:0' if self.cuda else 'cpu'
            checkpoint = torch.load(self.model_dir + load_name + '_model.pth', map_location=map_location)
            self.eval_vdn_net.load_state_dict(checkpoint['vdn_state_dict'])
            self.eval_rnn.load_state_dict(checkpoint['rnn_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Successfully load the model: {}".format(
                self.model_dir + load_name + '_model.pth'))
        else:
            raise Exception("No model!")

    def load_old_to_new(self,load_model_name,save_name):
        path_rnn = self.model_dir + load_model_name + 'rnn_net_params.pkl'
        path_vdn = self.model_dir + load_model_name + 'vdn_net_params.pkl'
        if os.path.exists(path_rnn):
            map_location = 'cuda:0' if self.cuda else 'cpu'
            torch.save({'vdn_state_dict': torch.load(path_vdn, map_location=map_location),
                        'rnn_state_dict': torch.load(
                path_rnn, map_location=map_location),
                        'optimizer_state_dict': self.optimizer.state_dict()},
                       self.model_dir + save_name + '_model.pth')
            print('Successfully load the model: {} and {}'.format(
                path_rnn, path_vdn))
            print('save to {}'.format(self.model_dir + save_name + '_model.pth'))
        else:
            raise Exception("No model!")