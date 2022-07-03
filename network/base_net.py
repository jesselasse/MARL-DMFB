import torch
import torch.nn as nn
import torch.nn.functional as f


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


class CRNN(nn.Module):
    def __init__(self, args):
        # design for fov=9
        super(CRNN, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1)
      # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
      # self.bn2 = nn.BatchNorm2d(32)
        self.mlp1 = nn.Linear(2+args.n_agents+args.n_actions, 10)
        self.rnn = nn.GRUCell(5*5*32+10, args.rnn_hidden_dim)
        self.fc1 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, inputs, hidden_state):
        pixel, vec = torch.split(
            inputs, [9*9*4, self.args.n_agents+self.args.n_actions+2], dim=1)
        pixel = pixel.reshape((-1, 4, 9, 9))
        pixel = f.relu(self.conv1(pixel))
        pixel = f.relu(self.conv2(pixel))
        pixel = pixel.reshape((-1, 5*5*32))  # (batch,800)
        vec = f.relu(self.mlp1(vec))  # (batch,10)
        x = torch.cat([pixel, vec], dim=1)  # (batch,810)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc1(h)
        return q, h


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
