# coding:utf-8

import random
from .agentbase import AgentBase
from .random_agent import RandomAgent
from .mcts_agent import MctsAgent

class MixedAgent(AgentBase):
    # いくつかのエージェントをランダムに呼び出して着手する
    def request_move(self, board, legal, player_num):
        now_agent = random.choice(self.agents)
        self.last_moved_agent = now_agent.get_agentname()
        return now_agent.request_move(board, legal, player_num)

    def get_agentname(self):
        return self.last_moved_agent

    def game_end(self, board, player_num, result):
        pass

    def __init__(self):
        self.agents = [RandomAgent(), MctsAgent(1000), MctsAgent(5000), MctsAgent(10000)]
        self.last_moved_agent = None    # 最後に着手したエージェントの名前