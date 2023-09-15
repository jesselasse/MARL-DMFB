import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

def conv_str(fov, id=3, od=32):
    conv1=nn.Conv2d(id, od, kernel_size=3, stride=1)
    conv2=nn.Conv2d(id, od, kernel_size=3, stride=2)
    conv3=nn.Conv2d(od, od, kernel_size=3, stride=1)
    convs={5:[conv1],
           7: [conv1, conv3],
           9: [conv1, conv3],
           11:[conv1, conv3],
           13:[conv1, conv3],
           19: [conv2, conv3, conv3]}
    return convs[fov]

class CRNN(nn.Module): #使用的网络
    def __init__(self, args,lod=10):
        super(CRNN, self).__init__()
        self.pixid=args.obs_shape[0]
        pixod=args.hyper_hidden_dim
        self.rnn_hidden_dim=args.rnn_hidden_dim
        self.n_actions=args.n_actions # 最终输出维度
        if args.last_action:
            self.lid = args.obs_shape[-2] + args.n_actions
        else:
            self.lid=args.obs_shape[-2]
        # difine conv
        self.convs=conv_str(args.fov, self.pixid, pixod)
      # self.bn2 = nn.BatchNorm2d(32)
        size=args.fov
        self.size=size
        i=1
        for conv in self.convs:
            self.add_module('conv{}'.format(i),conv)
            i+=1
            size=int((size+2*conv.padding[0]-conv.dilation[0]*(conv.kernel_size[0]-1)-1)//conv.stride[0]+1)
        self.rnnout=size*size*pixod #卷积神经网络输出大小
        # linear connection for vector input_dim=lid, output_dim=10
        self.mlp1 = nn.Linear(self.lid, lod)
        # gru
        self.rnn = nn.GRUCell(self.rnnout+lod, args.rnn_hidden_dim)
        self.fc1 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        # if args.attention:
        #     self.fc_attention = nn.Linear(2, 2)

    def forward(self, inputs, hidden_state):
        pixel, vec = torch.split(
            inputs, [inputs.shape[1]-self.lid, self.lid], dim=1)  # 分成FOV*FOV*3, n_actions+2
        pixel = pixel.reshape((-1, self.pixid, self.size, self.size))# (-1表示自动计算）（batchsize*n_agents,?,fov,fov)
        for conv in self.convs:
            pixel=f.relu(conv(pixel))
        pixel = pixel.reshape((-1, self.rnnout))  # (batchsize*n_agents,800) (800=5*5*32)
        vec = f.relu(self.mlp1(vec))  # (batchsize*n_agents,10)
        x = torch.cat([pixel, vec], dim=1)  # (batchsize*n_agents,810)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim) # (batchsize*n_agents, rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc1(h)
        return q, h


# class CRNN(nn.Module):
#     def __init__(self, args):
#         # design for fov=9
#         super(CRNN, self).__init__()
#         self.args = args
#         self.conv2 = None
#         self.conv3 = None
#         self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1)
#         self.size=3
#       # self.bn1 = nn.BatchNorm2d(32)
#         if self.args.fov>5:
#             self.conv2 = nn.Conv2d(24, 24, kernel_size=3, stride=1)
#         if self.args.fov > 7:
#             self.size=5
#         if self.args.fov > 10:
#             self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2)
#             self.conv3 = nn.Conv2d(24, 24, kernel_size=3, stride=1)
#       # self.bn2 = nn.BatchNorm2d(32)
#         self.mlp1 = nn.Linear(2+args.n_actions, 10)
#         self.rnn = nn.GRUCell(self.size*self.size*24+10, args.rnn_hidden_dim)
#         self.fc1 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
#
#     def forward(self, inputs, hidden_state):
#         pixel, vec = torch.split(
#             inputs, [self.args.fov*self.args.fov*3, self.args.n_actions+2], dim=1)  # FOV尺寸？ 分成FOV*FOV*4, n_agents+n_actions+2
#         pixel = pixel.reshape((-1, 3, self.args.fov, self.args.fov))
#         pixel = f.relu(self.conv1(pixel))
#         if self.conv2:
#             pixel = f.relu(self.conv2(pixel))
#         if self.conv3:
#             pixel = f.relu(self.conv3(pixel))
#         pixel = pixel.reshape((-1, self.size*self.size*24))  # (batch,800) ?这个数是啥
#         vec = f.relu(self.mlp1(vec))  # (batch,10)
#         x = torch.cat([pixel, vec], dim=1)  # (batch,810)
#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         h = self.rnn(x, h_in)
#         q = self.fc1(h)
#         return q, h


# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q


