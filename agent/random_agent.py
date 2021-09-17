# coding:utf-8

import random
from .agentbase import AgentBase


class RandomAgent(AgentBase):
    def request_move(self, flat_board, legal_moves, is_first):
        return random.choice(legal_moves)
    
    def get_agentname(self):
        return "RandomAgent"

    def __init__(self):
        pass