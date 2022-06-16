# coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F


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

        # 3x3
        self.conv1 = nn.Conv2d(in_channels=channels_in, out_channels=channels_in, kernel_size=(
            3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(num_features=channels_in)

        # 3x3
        self.conv2 = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=(
            3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(num_features=channels_out)

        if channels_in != channels_out:
            self.shortcut = nn.Conv2d(
                channels_in, channels_out, kernel_size=(1, 1))


class ResNet(nn.Module):
    def output(self, h):
        h = self.fc(h)
        return h

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

        return self.output(h)

    def __init__(self, channels_num):
        super(ResNet, self).__init__()

        self.conv = nn.Conv2d(in_channels=channels_num, out_channels=64, kernel_size=(
            5, 5), padding=(2, 2)).cuda()
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
        self.fc = nn.Linear(in_features=512, out_features=81).cuda()
        self.out = nn.Linear(
            in_features=81, out_features=1).cuda()  # valueでのみ使用


class PolicyNetwork(ResNet):
    def __init__(self, channels_num):
        super(PolicyNetwork, self).__init__(channels_num)


class ValueNetwork(ResNet):
    def output(self, h):
        h = self.fc(h)
        h = self.out(h)
        h = torch.tanh(h)
        return h

    def __init__(self, channels_num):
        super(ValueNetwork, self).__init__(channels_num)


def make_policynetwork():
    CHANNELS_NUM = 5
    model = PolicyNetwork(CHANNELS_NUM)
    return model


def make_valuenetwork():
    CHANNELS_NUM = 5
    model = ValueNetwork(CHANNELS_NUM)
    return model


class PVNetwork(nn.Module):
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
        p = self.policy_conv(h)
        p = self.policy_bn(p)
        p = p.reshape(p.size(0), -1)
        p = self.policy_out(p)

        v = self.value_conv(h)
        v = self.value_bn(v)
        v = v.reshape(v.size(0), -1)
        v = self.value_out(v)
        v = torch.tanh(v)

        return p, v


    def __init__(self, channels_num):
        super(PVNetwork, self).__init__()

        self.conv = nn.Conv2d(in_channels=channels_num, out_channels=32, kernel_size=(5, 5), padding=(2, 2))
        self.bn = nn.BatchNorm2d(32)

        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.block3 = Block(32, 32)
        self.block4 = Block(32, 64)
        self.block5 = Block(64, 64)
        self.block6 = Block(64, 64)
        self.block7 = Block(64, 128)
        self.block8 = Block(128, 128)

        self.globAvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.policy_conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1))
        self.policy_bn = nn.BatchNorm2d(num_features=128)
        self.policy_out = nn.Linear(in_features=128, out_features=81)
        self.value_conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1))
        self.value_bn = nn.BatchNorm2d(num_features=128)
        self.value_out = nn.Linear(in_features=128, out_features=1)

def make_pvnetwork(device="cuda"):
    CHANNELS_NUM = 5
    model = PVNetwork(CHANNELS_NUM).to(device)
    return model