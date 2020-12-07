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

    def ucb_move(self, board_num, legal, game_tree, win_rate):
        # UCBで手を判断する
        move, max_value = -1, -1.0
        total_num = sum(
            [win_rate[game_tree[board_num][pos]][1] for pos in legal])
        for pos in legal:
            next_board_num = game_tree[board_num][pos]
            win, num = win_rate[next_board_num]
            value = win / (num + 1.0) + math.sqrt(2 *
                                                  math.log(total_num + 1.0) / (num + 1.0))
            if max_value < value:
                move, max_value = pos, value
        return move

    def best_move(self, board_num, legal, game_tree, win_rate):
        # シミュレーションでもっとも選択された手を選択する
        move, max_num = -1, -1
        for pos in legal:
            next_board_num = game_tree[board_num][pos]
            _, num = win_rate[next_board_num]
            if max_num < num:
                move, max_num = pos, num
        return move

    def request_move(self, board, legal, player_num):
        # 現在の盤面から決まった回数プレイアウトして、一番良い手を返す
        root_board = Board(board)
        # game_tree[board_num][pos] = board_numの盤面で手番プレイヤーがposに打ったときに遷移するboardの番号
        game_tree = [[-1] * 81 for _ in range(60000)]
        # win_rate[board_num] = (そのboardに遷移する手を打ったときの勝率, シミュレーションでそのboardにたどり着いた回数)
        win_rate = [(0, 0)] * 60000
        # visited_num[board_num] = そのboardを訪れた回数
        visited_num = [0] * 60000
        board_count = 1
        for _ in range(self.playout_num):
            # プレイアウトしてrecordを作る
            now_board = copy.deepcopy(root_board)
            record = [(-1, -1, player_num)]
            now_player_num = player_num
            now_board_num, now_state = 0, 0
            random_selected = False
            while now_state == 0:
                now_legal = legal if now_board_num == 0 else now_board.legal_moves(
                    record[-1][1])
                move = -1
                if now_board_num != 0 and (random_selected or visited_num[now_board_num] < self.THRESHOLD):
                    move = self.random_move(now_legal)
                    random_selected |= True
                else:
                    if visited_num[now_board_num] == (0 if now_board_num == 0 else self.THRESHOLD):
                        # 展開する
                        for pos in range(81):
                            game_tree[now_board_num][pos] = board_count
                            board_count += 1
                    move = self.ucb_move(
                        now_board_num, now_legal, game_tree, win_rate)
                now_board.mark(move, now_player_num)
                if not random_selected:
                    record.append((now_board_num, move, now_player_num))
                    now_board_num = game_tree[now_board_num][move]
                now_player_num = now_player_num ^ 3
                now_state = now_board.check_state()

            # win_rateを更新する
            visited_num[0] += 1
            for board_num, pos, player in record:
                if board_num < 0:
                    continue
                next_board_num = game_tree[board_num][pos]
                visited_num[next_board_num] += 1
                win, num = win_rate[next_board_num]
                win_rate[next_board_num] = (
                    win + 1, num + 1) if now_state == player else (win, num + 1)
        return self.best_move(0, legal, game_tree, win_rate)

    def game_end(self, board, player_num, result):
        pass

    def __init__(self, playout_num):
        self.playout_num = playout_num
        self.THRESHOLD = 10
