import torch.nn as nn
import torch


class VDNNet(nn.Module):
    def __init__(self):
        super(VDNNet, self).__init__()

    def forward(self, q_values, dim=2):  # dim: agents dimmension
        return torch.nanmean(q_values, dim=dim,keepdim=True)
       # return torch.sum(q_values, dim=2, keepdim=True)
