# coding:utf-8

import random
from agentbase import AgentBase


class RandomAgent(AgentBase):
    def request_move(self, board, legal, player_num):
        return random.choice(legal)

    def game_end(self, board, player_num, result):
        pass

    def __init__(self):
        pass
