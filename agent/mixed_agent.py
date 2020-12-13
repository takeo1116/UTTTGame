# coding:utf-8

import random
from .agentbase import AgentBase
from .agent_util import constract_agent

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
        self.AGENT_NAMES = ["RandomAgent", "MctsAgent_1000", "MctsAgent_5000", "MctsAgent_10000"]
        self.agents = [constract_agent(agent_name) for agent_name in self.AGENT_NAMES]
        self.last_moved_agent = None    # 最後に着手したエージェントの名前