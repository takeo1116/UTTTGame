# coding:utf-8

import random
import torch
from torch import nn, optim
from .agentbase import AgentBase
from learning.learning_util import convert_board, make_network


class SupervisedLearningAgent(AgentBase):

    def request_move(self, board, legal, player_num):
        normal_board = board if player_num == 1 else [[0, 2, 1][mark] for mark in board]
        board_data = convert_board(normal_board, legal)
        inputs = torch.Tensor(board_data)
        outputs = self.model(inputs)
        _, move = torch.max(outputs.data, 0)
        if move not in legal:
            print("illegal move!")
        return move

    def game_end(self, board, player_num, result):
        pass

    def __init__(self, model_path):
        self.model = make_network()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
