# coding:utf-8

import torch
from torch import nn, optim


def convert_board(board, legal):
    # boardとlegalを学習、推論に適した形に変換する
    my_board = [1 if mark == 1 else 0 for mark in board]
    op_board = [1 if mark == 2 else 0 for mark in board]
    legal_board = [1 if pos in legal else 0 for pos in range(81)]
    board_data = my_board + op_board + legal_board
    return board_data


def convert_record(record):
    # recordを加工して、学習に適した教師データに変換する
    board, legal, move = record["board"], record["legal"], record["move"]
    board_data = convert_board(board, legal)
    return (board_data, move)


def make_network():
    # モデルを作る
    model = nn.Sequential()
    model.add_module("fc1", nn.Linear(81*3, 100))
    model.add_module("relu1", nn.ReLU())
    model.add_module("fc2", nn.Linear(100, 100))
    model.add_module("relu2", nn.ReLU())
    model.add_module("fc3", nn.Linear(100, 81))

    return model
