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
            value = win / (num + 1.0) + math.sqrt(2 *
                                                  math.log(total_num + 1.0) / (num + 1.0))
            if max_value < value:
                move, max_value = pos, value
        return move

    def frequent_move(self, board_num, legal):
        # シミュレーションでもっとも訪れた手を選択する
        move, max_num = -1, -1
        for pos in legal:
            next_board_num = self.game_tree[board_num][pos]
            _, num = self.win_rate[next_board_num]
            if max_num < num:
                move, max_num = pos, num
        return move

    def request_move(self, board, legal, player_num):
        # 現在の盤面から決まった回数プレイアウトして、一番良い手を返す
        root_board = Board(board)

        for _ in range(self.playout_num):
            # プレイアウトしてrecordを作る
            now_board = copy.deepcopy(root_board)
            record = [(0, -1, player_num)]
            now_player_num = player_num
            now_board_num, now_state = 0, 0
            board_count = 1
            random_selected = False
            while now_state == 0:
                now_legal = now_board.legal_moves(record[-1][1])
                move = -1
                simurated_num = self.win_rate[now_board_num][1]
                if now_board_num != 0 and (random_selected or simurated_num < self.THRESHOLD):
                    move = self.random_move(now_legal)
                    random_selected |= True
                else:
                    if simurated_num == 0 if now_board_num == 0 else self.THRESHOLD:
                        # 展開する
                        for pos in range(81):
                            self.game_tree[now_board_num][pos] = board_count
                            board_count += 1
                    move = self.ucb_move(now_board_num, now_legal)
                now_board.mark(move, now_player_num)
                if not random_selected:
                    record.append((now_board_num, move, now_player_num))
                    now_board_num = self.game_tree[now_board_num][move]
                now_player_num = now_player_num ^ 3
                now_state = now_board.check_state()

            # win_rateを更新する
            for board_num, pos, player in record:
                next_board_num = self.game_tree[board_num][pos]
                win, num = self.win_rate[next_board_num]
                self.win_rate[next_board_num] = (
                    win + 1, num + 1) if now_state == player else (win, num + 1)
        return self.frequent_move(0, legal)

    def game_end(self, board, player_num, result):
        pass

    def __init__(self, playout_num):
        self.playout_num = playout_num
        self.THRESHOLD = 10
        self.max_board_num = 1
        # game_tree[board_num][pos] = board_numの盤面で手番プレイヤーがposに打ったときに遷移するboardの番号
        self.game_tree = [[-1]*81]*10000
        # win_rate[board_num] = (そのboardに遷移する手を打ったときの勝率, シミュレーションでそのboardにたどり着いた回数)
        self.win_rate = [(0, 0)]*10000
