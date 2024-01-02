import numpy as np
import torch
from torch.distributions import Categorical


# Agent no communication
class Agentsp:
    def __init__(self, args, model_dir, nargs=None, targs=None):
        self.type=0
        self.n_actions = args.n_actions
        # self.n_agents = args.n_agents
        #self.n_agents = 8
        # self.obs_shape = args.obs_shape
        if nargs is None:
            return
        if args.alg == 'vdn':
            from policy.vdn import VDN
            self.policy = VDN(nargs, model_dir, targs, cuda=args.cuda)
        elif args.alg == 'qmix':
            from policy.qmix import QMIX
            self.policy = QMIX(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        self.nargs = nargs

    def choose_action(self, obs, avail_actions, epsilon, last_action=None, hidden_state=None):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        # agent_id = np.zeros(self.n_agents-1)
        # agent_id[0] = 1.
        # if self.nargs.last_action:
        #     if last_action is None:
        #         last_action = np.zeros(self.n_actions)
        #     inputs = np.hstack((inputs, last_action))
        # if self.args.reuse_network:
        #     inputs = np.hstack((inputs, agent_id))
        if hidden_state is None:
            hidden_state = torch.zeros((1, 1, self.nargs.rnn_hidden_dim))
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        q_value, hidden_state = self.policy.eval_rnn(inputs, hidden_state)
        # choose action from q value
        # q_value[avail_actions == 0.0] = - float("inf")
        if np.random.uniform() < epsilon: #jc改
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            action = torch.argmax(q_value)
        return action, hidden_state

    def choose_action4all(self, inputs, avail_actions, epsilon, hidden_state=None):
        valid_actions_indices = [np.nonzero(row)[0] for row in np.array(avail_actions)]

        n_agent, feat = inputs.shape
        if hidden_state is None:
            hidden_state = torch.zeros((1, n_agent, self.nargs.rnn_hidden_dim))
        inputs=torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()
        # get q value
        q_value, hidden_state = self.policy.eval_rnn(inputs, hidden_state)
        # choose action from q value
        q_value[avail_actions == 0.0] = - float("inf")
        q_value= q_value.squeeze()
        if np.random.uniform() < epsilon: #jc改
            action = [np.random.choice(indices) if len(indices) > 0 else -1 for indices in valid_actions_indices] # action是一个整数
        else:
            action = torch.argmax(q_value, dim=-1).tolist()
            if isinstance(action, int):
                action=[action]
        return action, hidden_state


    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len==0:
            return self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                if key == 'o':
                    batch[key] = batch[key][:, :max_episode_len+1]
                else:
                    batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        # if train_step > 0 and train_step % self.args.save_cycle == 0:
        #     self.policy.save_model(i,train_step)

class AgentT2(Agentsp):
    def __init__(self, *args):
        super(AgentT2, self).__init__(*args)
        self.type=1

    # def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
    #     avail_actions_ind = np.nonzero(avail_actions)[0]
    #     return np.random.choice(avail_actions_ind[1:4])

class Agentstore(Agentsp):
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.type=2

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        return 0, None

def Agents(args, task=0,model_dir=None):
    if task == 2:
        return Agentstore(args)
    nargs=args.netdata[task]
    nargs.obs_shape=args.obs_shape[task]
    nargs.load_model=args.load_model
    nargs.n_actions = args.n_actions
    if not model_dir:
        model_dir = args.model_dir + '/vdn/fov{}/task{}/'.format(nargs.fov, task)
    if hasattr(args, 'train'):
        targs=args.train
    else:
        targs = None
    if task == 0:
        return Agentsp(args, model_dir, nargs, targs)
    if task == 1:
        return AgentT2(args, model_dir, nargs, targs)
    else:
        raise TypeError('wrong agent task type')