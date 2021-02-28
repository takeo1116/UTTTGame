# coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F

# class Network(nn.Module):
#     def forward(self, x):
#         # 盤面と合法手を分けて、盤面はCNNに、合法手はそのまま出力につなぐ
#         b = F.relu(self.conv1(x))
#         b = F.relu(self.conv2(b))
#         b = F.relu(self.conv3(b))
#         b = F.relu(self.conv4(b))
#         b = F.relu(self.conv5(b))
#         b = F.relu(self.conv6(b))
#         b = F.relu(self.conv7(b))
#         b = F.relu(self.conv8(b))
#         b = F.relu(self.conv9(b))
#         b = F.relu(self.conv10(b))
        
#         h = b.view(-1, self.channels_num * 81)
#         h = F.relu(self.fc1(h))
#         return h
    

#     def __init__(self, channels_num):
#         super(Network, self).__init__()
#         self.channels_num = channels_num

#         self.conv1 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
#         self.conv2 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
#         self.conv3 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
#         self.conv4 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
#         self.conv5 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
#         self.conv6 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
#         self.conv7 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
#         self.conv8 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
#         self.conv9 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
#         self.conv10 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
#         self.fc1 = nn.Linear(in_features=channels_num * 9 * 9, out_features=9 * 9)

class Network(nn.Module):
    def forward(self, x):
        # 盤面と合法手を分けて、盤面はCNNに、合法手はそのまま出力につなぐ
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
        
        h = b.view(-1, self.channels_num * 81)
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
        self.conv6 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
        self.conv7 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
        self.conv8 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
        self.conv9 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
        self.conv10 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=(3, 3), padding=(1, 1))
        self.fc1 = nn.Linear(in_features=channels_num * 9 * 9, out_features=9 * 9)

# class Network(nn.Module):
#     def forward(self, x):
#         # 盤面と合法手を分けて、盤面はCNNに、合法手はそのまま出力につなぐ
#         b, l = torch.split(x, [self.channels_num-1, 1], dim=1)
#         b = F.relu(self.conv1(b))
#         b = F.relu(self.conv2(b))
#         b = F.relu(self.conv3(b))
#         b = F.relu(self.conv4(b))
#         b = F.relu(self.conv5(b))
#         b = F.relu(self.conv6(b))
#         b = F.relu(self.conv7(b))
#         b = F.relu(self.conv8(b))
#         b = F.relu(self.conv9(b))
#         b = F.relu(self.conv10(b))
#         h = torch.cat((b, l), dim=1)
        
#         h = h.view(-1, self.channels_num * 81)
#         h = F.relu(self.fc1(h))
#         return h
    

#     def __init__(self, channels_num):
#         super(Network, self).__init__()
#         self.channels_num = channels_num

#         self.conv1 = nn.Conv2d(in_channels=channels_num-1, out_channels=channels_num-1, kernel_size=(3, 3), padding=(1, 1))
#         self.conv2 = nn.Conv2d(in_channels=channels_num-1, out_channels=channels_num-1, kernel_size=(3, 3), padding=(1, 1))
#         self.conv3 = nn.Conv2d(in_channels=channels_num-1, out_channels=channels_num-1, kernel_size=(3, 3), padding=(1, 1))
#         self.conv4 = nn.Conv2d(in_channels=channels_num-1, out_channels=channels_num-1, kernel_size=(3, 3), padding=(1, 1))
#         self.conv5 = nn.Conv2d(in_channels=channels_num-1, out_channels=channels_num-1, kernel_size=(3, 3), padding=(1, 1))
#         self.conv6 = nn.Conv2d(in_channels=channels_num-1, out_channels=channels_num-1, kernel_size=(3, 3), padding=(1, 1))
#         self.conv7 = nn.Conv2d(in_channels=channels_num-1, out_channels=channels_num-1, kernel_size=(3, 3), padding=(1, 1))
#         self.conv8 = nn.Conv2d(in_channels=channels_num-1, out_channels=channels_num-1, kernel_size=(3, 3), padding=(1, 1))
#         self.conv9 = nn.Conv2d(in_channels=channels_num-1, out_channels=channels_num-1, kernel_size=(3, 3), padding=(1, 1))
#         self.conv10 = nn.Conv2d(in_channels=channels_num-1, out_channels=channels_num-1, kernel_size=(3, 3), padding=(1, 1))
#         self.fc1 = nn.Linear(in_features=channels_num * 9 * 9, out_features=9 * 9)