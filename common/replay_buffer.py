import numpy as np
import threading
import torch


class ReplayBuffer:
    def __init__(self, args, episode_limit):
        self.args = args
        self.n_actions = self.args.n_actions
        self.obs_shape = self.args.obs_shape[args.task]
        self.size = self.args.train.buffer_size
        # memory management
        self.buffers={}
        self.episode_limit = episode_limit
        #self.buffers={n_agents: [self._init_buffer(n_agents), 0, 0]} #分别代表对应n_agents的buffer, current_size, current_idx

        # thread lock
        self.lock = threading.Lock()

    def _init_buffer(self, n_agents,shape):
        # create the buffer to store info；； ‘o-->'s_c' and 's_d' for chip state and droplets state
        buffer = {#'s_c': torch.zeros((self.size,)+shape1), #(o要多一个，最后一步走完之后的状态）
                  'o': torch.zeros((self.size, self.episode_limit+1, n_agents, shape)),
                        'u': torch.zeros(self.size, self.episode_limit, n_agents, 1, dtype=torch.int8),
                        'r': torch.zeros(self.size, self.episode_limit, 1),
                        'avail_u': torch.zeros(self.size, self.episode_limit, n_agents, self.n_actions, dtype=bool),
                        'padded': torch.ones(self.size, self.episode_limit, 1, dtype=bool),
                        'terminated': torch.ones(self.size, self.episode_limit, 1, dtype=bool)
                        }
        if self.args.alg == 'maven':
            buffer['z'] = np.empty([self.size, self.args.noise_dim])
        return buffer

    # store the episode
    def store_episode(self, episodes):
        n_agents = episodes[0]['u'].shape[2]

        # 如果n_agents个数不在self.buffers的key里，就新建一个key,obs尺寸就在这里初始化。
        if n_agents not in self.buffers.keys():
            # # if env changes with time
            # shape1 = episodes[0]['s_c'].shape[1:]
            # shape2 = episodes[0]['s_d'].shape[-1]
            # if shape1[0]==episodes[0]['s_d'].shape[1]:
            #     shape1 = (self.self.episode_limit+1, )+shape1[1:]
            self.buffers[n_agents] = [self._init_buffer(n_agents, episodes[0]['o'].shape[-1]),0,0]
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
                # try:
                #     buffer['s_c'][idx][:episode_len + 1, :, :] = episodes[i]['s_c']
                # except:
                #     buffer['s_c'][idx] = episodes[i]['s_c']
                buffer['o'][idx][:episode_len+1,:,:] = episodes[i]['o']
                buffer['u'][idx][:episode_len,:,:] = episodes[i]['u']
                buffer['r'][idx][:episode_len,:] = episodes[i]['r']
                buffer['avail_u'][idx][:episode_len,:,:] = episodes[i]['avail_u']
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


