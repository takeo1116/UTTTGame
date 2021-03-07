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
        inputs = torch.Tensor([board_data]).cuda()
        outputs = self.model(inputs).cuda()
        scores = outputs[0].data.tolist()
        nowScore, move = -1.0, -1
        for idx, score in enumerate(scores):
            if score > nowScore and idx in legal:
                move = idx
                nowScore = score
        if move not in legal:
            print("illegal move!")
        # print(legal, move)
        return move

    def get_agentname(self):
        return "SupervisedLearningAgent"

    def game_end(self, board, player_num, result):
        pass

    def __init__(self, model_path):
        self.model = make_network()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
