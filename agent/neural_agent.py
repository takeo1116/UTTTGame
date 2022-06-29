# coding:utf-8

import random
import copy
import math
import torch
from .agentbase import AgentBase
from engine.board import Board
from learn.util.feature import make_feature
from learn.util.pickmove import pick_legalmoves

class NeuralAgent(AgentBase):
    def request_move(self, flat_board, legal_moves, is_first):
        feature = make_feature(flat_board, legal_moves)
        feature_tensor = torch.Tensor([feature]).to(self.device)
        p, _ = self.model(feature_tensor)
        moves = pick_legalmoves(p, [legal_moves], self.temperature)
        return moves[0]

    def get_agentname(self):
        return "NeuralAgent"

    def __init__(self, model, temperature=1.0, device="cpu"):
        self.THRESHOLD = 10
        self.model = model
        self.temperature = temperature
        self.device = device
        self.model.eval()