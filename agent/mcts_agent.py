# coding:utf-8

import random
import copy
from .agentbase import AgentBase
from engine.board import Board

class MctsAgent(AgentBase):
    def request_move(self, board, legal, player_num):
        # 現在の盤面から決まった回数プレイアウトして、一番良い手を返す
        root_board = Board(board)
        
        # とりあえず100回プレイアウト
        for play in range(100):
            now_board = copy.deepcopy(root_board)
            
            # 

        return random.choice(legal)

    def game_end(self, board, player_num, result):
        pass

    def __init__(self):
        pass
