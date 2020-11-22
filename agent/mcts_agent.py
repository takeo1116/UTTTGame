# coding:utf-8

import random
import copy
import math
from .agentbase import AgentBase
from .random_agent import RandomAgent
from engine.board import Board


class MctsAgent(AgentBase):
    def random_move(self, legal):
        return random.choice(legal)

    def ucb_move(self, board_num, legal):
        # UCBで手を判断する
        move, max_value = -1, -1.0
        total_num = sum(
            [self.win_rate[self.game_tree[board_num][pos]][1] for pos in legal])
        for pos in legal:
            next_board_num = self.game_tree[board_num][pos]
            win, num = self.win_rate[next_board_num]
            value = win/(num + 1.0) + math.sqrt(2 * math.log(total_num) / num)
            if value > max_value:
                move, max_value = pos, value
        return move

    def request_move(self, board, legal, player_num):
        # 現在の盤面から決まった回数プレイアウトして、一番良い手を返す
        root_board = Board(board)

        # とりあえず100回プレイアウト
        for play in range(100):
            # プレイアウトしてrecordを作る
            now_board = copy.deepcopy(root_board)
            record = []
            now_player_num = player_num
            now_board_num, now_state = 0, 0
            while now_state == 0:
                now_legal_moves = now_board.legal_moves()
                if win_rate[now_board_num][1] <= self.THRESHOLD:
                    # ランダムで選択
                    pass
                else:
                    # UCBで選択
                    pass
                now_player_num = now_player_num ^ 3
                now_state = now_board.check_state()

            # 更新する

        return random.choice(legal)

    def game_end(self, board, player_num, result):
        pass

    def __init__(self):
        self.THRESHOLD = 10
        self.max_board_num = 1
        self.game_tree = [[-1]*81]*10000
        self.win_rate = [(0, 0)]*10000
