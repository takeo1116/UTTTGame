# coding:utf-8

import random
from .agentbase import AgentBase
from .random_agent import RandomAgent
from .mcts_agent import MctsAgent


class MixedAgent(AgentBase):
    # いくつかのエージェントをランダムに呼び出して着手する
    def request_move(self, flat_board, legal_moves, is_first):
        now_agent = random.choice(self.agents)
        self.last_moved_agent = now_agent.get_agentname()
        return now_agent.request_move(flat_board, legal_moves, is_first)

    def get_agentname(self):
        return self.last_moved_agent

    def __init__(self):
        self.agents = [RandomAgent(), MctsAgent(
            1000), MctsAgent(5000), MctsAgent(10000)]
        self.last_moved_agent = "MixedAgent"    # 最後に行動したエージェントの名前
