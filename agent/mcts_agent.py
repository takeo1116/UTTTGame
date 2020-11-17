# coding:utf-8

import random
import copy
from .agentbase import AgentBase
from .random_agent import RandomAgent
from engine.board import Board

class MctsAgent(AgentBase):
    def request_move(self, board, legal, player_num):
        # 現在の盤面から決まった回数プレイアウトして、一番良い手を返す
        root_board = Board(board)
        random_agent = RandomAgent()
        max_board_num = 1
        game_tree = [[0]*81]*10000
        win_rate = [(0, 0)]*10000

        # とりあえず100回プレイアウト
        for play in range(100):
            now_board = copy.deepcopy(root_board)
            record = []
            now_player_num = player_num
            now_board_num, now_state = 0, 0
            while now_state == 0:
                now_flat_board = now_board.flatten()
                now_legal_moves = now_board.legal_moves()
                move = random_agent.request_move(now_flat_board, now_legal_moves, now_player_num)

                now_player_num = now_player_num ^ 3
                now_state = now_board.check_state()
            


        return random.choice(legal)

    def game_end(self, board, player_num, result):
        pass

    def __init__(self):
        pass
