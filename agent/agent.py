import numpy as np
import torch
from torch.distributions import Categorical


# Agent no communication
class Agentsp:
    def __init__(self, args, nargs=None):
        self.n_actions = args.n_actions
        # self.n_agents = args.n_agents
        #self.n_agents = 8
        # self.obs_shape = args.obs_shape
        if nargs is None:
            nargs=args
        if args.alg == 'vdn':
            from policy.vdn import VDN
            self.policy = VDN(args, nargs)
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
        if self.nargs.last_action:
            if last_action is None:
                last_action = np.zeros(self.n_actions)
            inputs = np.hstack((inputs, last_action))
        # if self.args.reuse_network:
        #     inputs = np.hstack((inputs, agent_id))
        if hidden_state is None:
            hidden_state = torch.zeros(self.nargs.rnn_hidden_dim)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        q_value, hidden_state = self.policy.eval_rnn(inputs, hidden_state)
        # choose action from q value
        q_value[avail_actions == 0.0] = - float("inf")
        if np.random.uniform() < epsilon: #jc改
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            action = torch.argmax(q_value)
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

    # def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
    #     avail_actions_ind = np.nonzero(avail_actions)[0]
    #     return np.random.choice(avail_actions_ind[1:4])

class Agentstore(Agentsp):
    def __init__(self, *args):
        super().__init__(*args)

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        return 0, None

def Agents(*args, task=0):
    if task == 0:
        return Agentsp(*args)
    if task == 1:
        return AgentT2(*args)
    if task == 2:
        return Agentstore(*args)
    else:
        raise TypeError('wrong agent task type')