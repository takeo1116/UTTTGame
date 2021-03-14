# coding:utf-8

from .board import Board
from agent.agent_util import constract_agent
import time


class GameParallel:
    # 複数のゲームをまとめて処理する（強化学習用）
    def get_nowboards(self):
        # 次に打つプレイヤーと現在の盤面を取得する
        # 全ゲームが終了していたらflat_boardsが空リストになる
        def get_prevmove(idx):
            return -1 if len(self.game_records[idx]) <= 0 else self.game_records[idx][-1][4]
        board_infos = [(self.boards[idx].flatten(), self.boards[idx].legal_moves(get_prevmove(idx)))
                       for idx, state in enumerate(self.game_states) if self.boards[idx].check_state() == 0]
        return (self.now_player, board_infos)

    def process_games(self, moves):
        # 手の一覧を入力することで盤面を進める
        processing_boards = [(idx, board) for idx, board in enumerate(
            self.boards) if board.check_state() == 0]
        if len(processing_boards) != len(moves):
            print("illegal size of moves")
            return
        for (idx, board), move in zip(processing_boards, moves):
            # 反則手があればそのゲームを終了し、recordを書き換える
            prev_move = -1
            if len(self.game_records[idx]) > 0:
                _, _, _, _, prev_move = self.game_records[idx][-1]
            legal_moves = board.legal_moves(prev_move)
            if move in legal_moves:
                board.mark(move, self.now_player)
                self.game_states[idx] = board.check_state()
                self.game_records[idx].append(
                    (self.now_player, self.now_player, self.boards[idx].flatten(), legal_moves, move))
            else:
                self.game_states[idx] = self.now_player ^ 3
                self.game_records[idx].clear()
                self.game_records[idx].append(
                    (self.now_player, self.now_player, self.boards[idx].flatten(), legal_moves, move))
        self.now_player = self.now_player ^ 3
        return True

    def __init__(self, parallel_num):
        self.parallel_num = parallel_num
        self.boards = [Board() for _ in range(parallel_num)]
        self.now_player = 1
        # 0:ゲーム進行中, 1:player1の勝ち, 2:player2の勝ち, 3:draw
        self.game_states = [0 for _ in range(parallel_num)]
        self.game_records = [[] for _ in range(parallel_num)]
