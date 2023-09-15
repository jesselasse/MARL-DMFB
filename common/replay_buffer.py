import numpy as np
import threading
import torch


class ReplayBuffer:
    def __init__(self, args, n_agents=1):
        self.args = args
        self.n_actions = self.args.n_actions
        self.obs_shape = self.args.obs_shape[-1]
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        # memory management
        self.buffers={}
        #self.buffers={n_agents: [self._init_buffer(n_agents), 0, 0]} #分别代表对应n_agents的buffer, current_size, current_idx

        # thread lock
        self.lock = threading.Lock()

    def _init_buffer(self, n_agents):
        # create the buffer to store info
        buffer = {'o': torch.zeros((self.size, self.episode_limit+1, n_agents, self.obs_shape)), #(o要多一个，最后一步走完之后的状态）
                        'u': torch.zeros(self.size, self.episode_limit, n_agents, 1, dtype=torch.int8),
                        'r': torch.zeros(self.size, self.episode_limit, 1),
                        'avail_u': torch.zeros(self.size, self.episode_limit, n_agents, self.n_actions, dtype=bool),
                        'u_onehot': torch.zeros(self.size, self.episode_limit, n_agents, self.n_actions, dtype=bool),
                        'padded': torch.ones(self.size, self.episode_limit, 1, dtype=bool),
                        'terminated': torch.ones(self.size, self.episode_limit, 1, dtype=bool)
                        }
        if self.args.alg == 'maven':
            buffer['z'] = np.empty([self.size, self.args.noise_dim])
        return buffer

    # store the episode
    def store_episode(self, episodes):
        # (删去）episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
        # 例：episodes[0]['o']=(本身episode_len+1, n_agents,具体维度)每个不一样： episode[0]和episode[1]的episode_len不一样
        # episode_batch = episodes[0]
        # episodes.pop(0)
        # for episode in episodes:
        #     for key in episode_batch.keys():
        #         episode_batch[key] = np.concatenate(
        #             (episode_batch[key], episode[key]), axis=0)
        # 判断episodes里有多少n_agents
        n_agents = episodes[0]['u'].shape[2]
        # 如果n_agents个数不在self.buffers的key里，就新建一个key
        if n_agents not in self.buffers.keys():
            self.buffers[n_agents] = [self._init_buffer(n_agents),0,0]
        # 如果self.buffers超过2个就把第一个删了
        if len(self.buffers) > 2:
            self.buffers.pop(list(self.buffers.keys())[0])
        buffer, current_size, current_idx = self.buffers[n_agents]

        batch_size = len(episodes)  # episode_number
        n_agents = episodes[0]['u'].shape[2]

        with self.lock:
            idxs, current_size, current_idx = self._get_storage_idx(current_size, current_idx, inc=batch_size)
            self.buffers[n_agents][1:] = [current_size, current_idx]
            for i in range(batch_size):
                idx=idxs[i]
                episode_len = episodes[i]['u'].shape[1]
                # store the informations
                buffer['o'][idx][:episode_len+1,:,:] = episodes[i]['o']
                buffer['u'][idx][:episode_len,:,:] = episodes[i]['u']
                buffer['r'][idx][:episode_len,:] = episodes[i]['r']
                # buffer['o_next'][idx] = episode_batch['o_next']
                buffer['avail_u'][idx][:episode_len,:,:] = episodes[i]['avail_u']
                # buffer['avail_u_next'][idxs] = episode_batch['avail_u_next']
                buffer['u_onehot'][idx][:episode_len,:,:] = episodes[i]['u_onehot']
                buffer['padded'][idx][:episode_len,:] = episodes[i]['padded']
                buffer['terminated'][idx][:episode_len,:] = episodes[i]['terminated']
                if self.args.alg == 'maven':
                    buffer['z'][idx]= episodes[i]['z']

    # 原先的store method
    def addto(self, episode_batch):
        batch_size = episode_batch['r'].shape[0]  # episode_number

        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['r'][idxs] = episode_batch['r']
            # self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            # self.buffers['avail_u_next'][idxs] = episode_batch['avail_u_next']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']
            if self.args.alg == 'maven':
                self.buffers['z'][idxs] = episode_batch['z']

    def sample(self, batch_size):

        current_sizes=[self.buffers[key][1] for key in self.buffers.keys()]
        total_size = sum(current_sizes)
        batch_size = min(total_size, batch_size)
        idx = np.random.choice(total_size, batch_size, replace=False) #修改 randint-->choice
        m = [0]+current_sizes
        idx_list=[]
        for i in range(len(current_sizes)):
            # idx 里每个数减1
            idx -= m[i]
            # 将idx拆分成小于m[i]和大于m[i]的两个变量
            idx_list.append(idx[idx < current_sizes[i]])
            idx= idx[idx >= current_sizes[i]]
        buffer_list=[]

        for i, n_agent in enumerate(self.buffers.keys()):
            temp_buffer = {}
            buffer = self.buffers[n_agent][0]
            idx=idx_list[i]
            if not idx.size:
                continue
            for key in buffer.keys():
                temp_buffer[key] = buffer[key][idx]
            temp_buffer['avail_u_next'] = buffer['avail_u'][idx]
            buffer_list.append(temp_buffer)
        return buffer_list


    def _get_storage_idx(self, current_size, current_idx, inc=None):
        inc = inc or 1
        if current_idx + inc <= self.size:
            idx = np.arange(current_idx, current_idx + inc)
            current_idx += inc
        elif current_idx < self.size:
            overflow = inc - (self.size - current_idx)
            idx_a = np.arange(current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            current_idx = overflow
        else:
            idx = np.arange(0, inc)
            current_idx = inc
        current_size = min(self.size, current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx, current_size, current_idx
