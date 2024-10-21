import torch
import torch.nn as nn
import torch.nn.functional as F

class Channel(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(Channel, self).__init__()
        self.gate_c = nn.Sequential()
        gate_channels = [gate_channel, gate_channel//reduction_ratio, gate_channel]

        self.gate_c.add_module('gate_c_fc_1', nn.Linear(gate_channels[0], gate_channels[1]))
        self.gate_c.add_module('gate_c_bn_1', nn.BatchNorm1d(gate_channels[1]))
        self.gate_c.add_module('gate_c_relu_1', nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))


    def forward(self, in_tensor):
        avg_pool = F.avg_pool1d(in_tensor, in_tensor.size(2), in_tensor.size(2))
        avg_pool=avg_pool.view(avg_pool.shape[0], avg_pool.shape[1])
        
        gate_c = self.gate_c(avg_pool)
        gate_c = torch.unsqueeze(gate_c, 2)
        gate_c = gate_c.expand_as(in_tensor)

        return gate_c

class Modulator(nn.Module):
    def __init__(self, args):
        super(Modulator, self).__init__()
        self.channel_att = Channel(gate_channel=args.gate_channels, reduction_ratio=args.reduction_ratio)

    def forward(self, in_tensor):
        att = self.channel_att(in_tensor)
        att = torch.sigmoid(att)

        return att * in_tensor

