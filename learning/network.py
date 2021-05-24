# coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear


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


class Block(nn.Module):

    def forward(self, x):
        h = self.conv1(x)
        h = F.relu(self.bn1(h))
        h = self.conv2(h)
        h = self.bn2(h)
        if self.channels_in != self.channels_out:
            shortcut = self.shortcut(x)
            y = F.relu(h + shortcut)
        else:
            y = F.relu(h + x)
        return y

    def __init__(self, channels_in, channels_out):
        super(Block, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        channels = channels_out // 4

        # 1x1
        self.conv1 = nn.Conv2d(in_channels=channels_in, out_channels=channels, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.bn1 = nn.BatchNorm2d(num_features=channels).cuda()

        # 3x3
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels_out, kernel_size=(3, 3), padding=(1, 1)).cuda()
        self.bn2 = nn.BatchNorm2d(num_features=channels_out).cuda()

        if channels_in != channels_out:
            self.shortcut = nn.Conv2d(channels_in, channels_out, kernel_size=(1, 1)).cuda()


class ResNet(nn.Module):
    def forward(self, x):
        h = self.conv(x)
        h = F.relu(self.bn(h))

        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.block8(h)

        h = self.globAvgPool(h)
        h = h.reshape(h.size(0), -1)
        h = self.fc(h)

        return h

    def __init__(self, channels_num):
        super(ResNet, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=channels_num, out_channels=64, kernel_size=(5, 5), padding=(2, 2)).cuda()
        self.bn = nn.BatchNorm2d(64).cuda()

        self.block1 = Block(64, 64)
        self.block2 = Block(64, 128)
        self.block3 = Block(128, 128)
        self.block4 = Block(128, 256)
        self.block5 = Block(256, 256)
        self.block6 = Block(256, 512)
        self.block7 = Block(512, 512)
        self.block8 = Block(512, 512)

        self.globAvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1)).cuda()
        self.fc = nn.Linear(in_features=512, out_features=9*9).cuda()
