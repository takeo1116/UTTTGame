# coding:utf-8

import torch
from torch import nn


def pick_legalmoves(outputs, legal_moves, temp=1.0, device="cpu"):
    # 合法手のなかからsoftmaxで手を選択する
    # temp > 0.0
    beta = 1.0/temp
    softmax = nn.Softmax(dim=1).to(device)
    raw_probs = softmax(beta * outputs)
    probs = torch.Tensor([[prob + 0.001 if idx in legal else 0.0 for idx, prob in enumerate(
        probs)] for probs, legal in zip(raw_probs.tolist(), legal_moves)])
    moves = torch.multinomial(probs, 1)
    moves = moves.flatten()
    return moves.tolist()
