# coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    def forward(self, x):
        # AlphaGOと同じ構成
        b = F.relu(self.conv1(x))
        b = F.relu(self.conv2(b))
        b = F.relu(self.conv3(b))
        b = F.relu(self.conv4(b))
        b = F.relu(self.conv5(b))
        b = F.relu(self.conv6(b))
        b = F.relu(self.conv7(b))
        b = F.relu(self.conv8(b))
        b = F.relu(self.conv9(b))
        b = F.relu(self.conv10(b))
        b = F.relu(self.conv11(b))
        b = self.conv12(b)
        b = self.conv13(b)
        
        h = b.view(-1, self.channels_num * 81)
        h = F.relu(self.fc1(h))
        return h
    
    def __init__(self, channels_num):
        super(Network, self).__init__()
        self.channels_num = channels_num
        self.kernel_coef = 30
        k = self.kernel_coef * channels_num
        self.k = k

        self.conv1 = nn.Conv2d(in_channels=channels_num, out_channels=k, kernel_size=(5, 5), padding=(2, 2)).cuda()
        self.conv2 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv3 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv4 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv5 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv6 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv7 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv8 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv9 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv10 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv11 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv12 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv13 = nn.Conv2d(in_channels=k, out_channels=channels_num, kernel_size=(1, 1)).cuda()
        self.fc1 = nn.Linear(in_features=channels_num * 9 * 9, out_features=9 * 9).cuda()


class ValueNetwork(nn.Module):
    def forward(self, x):
        b = F.relu(self.conv1(x))
        b = F.relu(self.conv2(b))
        b = F.relu(self.conv3(b))
        b = F.relu(self.conv4(b))
        b = F.relu(self.conv5(b))
        b = F.relu(self.conv6(b))
        b = F.relu(self.conv7(b))
        b = F.relu(self.conv8(b))
        b = F.relu(self.conv9(b))
        b = F.relu(self.conv10(b))
        b = F.relu(self.conv11(b))
        b = self.conv12(b)
        b = self.conv13(b)
        
        h = b.view(-1, self.channels_num * 81)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        h = torch.tanh(h)
        return h
    
    def __init__(self, channels_num):
        super(ValueNetwork, self).__init__()
        self.channels_num = channels_num
        self.kernel_coef = 30
        k = self.kernel_coef * channels_num
        self.k = k

        self.conv1 = nn.Conv2d(in_channels=channels_num, out_channels=k, kernel_size=(5, 5), padding=(2, 2)).cuda()
        self.conv2 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv3 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv4 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv5 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv6 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv7 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv8 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv9 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv10 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv11 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv12 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.conv13 = nn.Conv2d(in_channels=k, out_channels=channels_num, kernel_size=(1, 1)).cuda()
        self.fc1 = nn.Linear(in_features=channels_num * 9 * 9, out_features=256).cuda()
        self.fc2 = nn.Linear(in_features=256, out_features=1).cuda()
