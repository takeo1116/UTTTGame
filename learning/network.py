# coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(x))
        h = F.relu(self.conv3(x))
        h = F.relu(self.conv4(x))
        h = F.relu(self.conv5(x))
        
        h = h.view(-1, self.channels_num * 81)
        h = F.relu(self.fc1(h))
        return h

    def __init__(self, channels_num):
        super(Network, self).__init__()
        self.channels_num = channels_num

        self.conv1 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))

        self.fc1 = nn.Linear(in_features=channels_num * 9 * 9, out_features=9 * 9)