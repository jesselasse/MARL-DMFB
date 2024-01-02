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
        self.pixid=args.obs_shape[0] # 2
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
        self.rnnout=size*size*pixod #卷积神经网络输出大小 9*9为 800
        # linear connection for vector input_dim=lid, output_dim=10
        self.mlp1 = nn.Linear(self.lid, lod)
        # gru
        self.rnn = nn.GRUCell(self.rnnout+lod, args.rnn_hidden_dim)
        self.fc1 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        # if args.attention:
        #     self.fc_attention = nn.Linear(2, 2)

    def forward(self, inputs, hidden_state):
        batch, n_agent, _ =inputs.shape
        inputs=inputs.flatten(end_dim=1)
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
        q = q.view(batch, n_agent,-1)
        return q, h

class CRNN2(nn.Module): #使用的网络
    def __init__(self, args,lod=10):
        super(CRNN2, self).__init__()
        self.pixid=args.obs_shape[0] # 2
        pixod=args.hyper_hidden_dim
        self.rnn_hidden_dim=args.rnn_hidden_dim
        self.n_actions=args.n_actions # 最终输出维度
        if args.last_action:
            self.lid = args.obs_shape[-2] + args.n_actions
        else:
            self.lid=args.obs_shape[-2]
        # difine conv
        self.convs=conv_str(args.fov, self.pixid, pixod)
        self.convs2=conv_str(args.fov, 2, pixod)
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
        batch, n_agent, _ =inputs.shape
        inputs=inputs.flatten(end_dim=1)
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
        q = q.view(batch, n_agent,-1)
        return q, h

class FC4(nn.Module):

    def __init__(self, lid, n_actions, lod=16, atto=32, hd=32):
        # 先试一下一层？
        super(FC4, self).__init__()
        self.fc1 = nn.Linear(lid, lod)
        self.gru = nn.GRUCell(lod, hd)
        self.att1 = nn.Linear(lid,atto)
        self.att2 = nn.Linear(atto, atto)
        self.fc2 = nn.Linear(hd+atto, n_actions)

    def forward(self, inputs, hidden):
        # 这里的input就是‘s_d'（batch,n_agent,length)
        # 首先拆分自己的信息和相关性矩阵
        batch, n_agent, feat = inputs.shape
        feat = feat - n_agent
        self_vec = inputs[:, :, :feat]
        matrix = inputs[:, :, feat:]

        #自己的信息分别传入三个全连接
        hidden = self.gru(f.relu(self.fc1(self_vec)).flatten(end_dim=1), hidden.flatten(end_dim=1)).view(batch,n_agent,-1)
        att = f.relu(self.att2(f.relu(self.att1(self_vec))))
        att_vec = torch.bmm(matrix,att)
        x = torch.cat([hidden, att_vec], dim=-1)
        q = self.fc2(x)
        return q, hidden


class CRnnattFC1(nn.Module):

    def __init__(self, args, lod=64, atto=64):
        # lid 包括自己+其他所有相关的经过一层卷积后加权求和得到的向量
        # 两层好了
        super(CRnnattFC1, self).__init__()
        self.pixid = args.obs_shape[0]
        pixod = args.hyper_hidden_dim
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.n_actions = args.n_actions  # 最终输出维度
        if args.last_action:
            self.lid = args.obs_shape[3] + args.n_actions
        else:
            self.lid = args.obs_shape[3]
        # difine conv
        self.convs = conv_str(args.fov, self.pixid, pixod)
        # self.bn2 = nn.BatchNorm2d(32)
        size = args.fov
        self.size = size
        i = 1
        for conv in self.convs:
            self.add_module('conv{}'.format(i), conv)
            i += 1
            size = int(
                (size + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1)
        self.cnnout = size * size * pixod
        self.fc1 = nn.Linear(self.lid, lod)
        self.gru = nn.GRUCell(self.cnnout + lod, args.rnn_hidden_dim)
        self.att1 = nn.Linear(self.lid, atto)
        self.att2 = nn.Linear(atto, atto)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + atto, args.n_actions)

    def forward(self, inputs, hidden):
        batch, n_agent, feat = inputs.shape
        feat = feat - n_agent
        pix_vec = inputs[:, :, :feat]
        matrix = inputs[:, :, feat:]
        feat = feat-self.lid
        pix = pix_vec[:,:,:feat]
        vec = pix_vec[:, :, feat:]

        pixel = pix.flatten(end_dim=1).reshape((-1, self.pixid, self.size, self.size))
        for conv in self.convs:
            pixel=f.relu(conv(pixel))
        pixel = pixel.flatten(start_dim=1)
        hinput=torch.cat([pixel, f.relu(self.fc1(vec)).flatten(end_dim=1)], dim=1)

        # 自己的信息分别传入三个全连接
        hidden = self.gru(hinput, hidden.flatten(end_dim=1)).view(batch, n_agent,
                                                                                                         -1)
        att = f.relu(self.att2(f.relu(self.att1(vec))))
        att_vec = torch.bmm(matrix, att)
        x = torch.cat([hidden, att_vec], dim=-1)
        q = self.fc2(x)
        return q, hidden



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


