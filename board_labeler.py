# coding:utf-8

import os
import json
import concurrent.futures
from agent.random_agent import RandomAgent
from agent.mcts_agent import MctsAgent

class BoardLabeler():
    def __init__(self):
        self.agent_names = ["RandomAgent", "MctsAgent_1000", "MctsAgent_5000", "MctsAgent_10000"]
        