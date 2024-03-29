# coding:utf-8

from engine.record import MoveData, Record, RecordResult
from .board import Board
import time


class Game:
    def make_recordresult(self, game_state):
        return [RecordResult.NOSET, RecordResult.PLAYER1WIN, RecordResult.PLAYER2WIN, RecordResult.DRAW][game_state]

    def process_game(self):
        # ゲームを1手進める
        flat_board = self.board.flatten()
        legal_moves = self.board.legal_moves(self.prev_move)
        # now_playerに手を聞く
        # これを変更したくて、常に自分の石が1になるようにこの段階でしておく、先手かどうかを引数で入れる
        player = self.players[self.now_player - 1]
        is_first = (self.now_player == self.first_player)
        request_board = [mark for mark in flat_board] if self.now_player == 1 else [
            [0, 2, 1][mark] for mark in flat_board]
        move = player.request_move(
            request_board, legal_moves.copy(), is_first)
        # 帰ってきた手が有効かどうか調べる（有効じゃない手を打とうとしたら負け）
        if move not in legal_moves:
            self.game_state = self.now_player ^ 3
        # 手を反映させる
        agent_name = player.get_agentname()
        self.board.mark(move, self.now_player)
        self.record.append(MoveData(self.now_player, is_first,
                                    agent_name, request_board, legal_moves, move))
        self.prev_move = move
        # ゲーム終了かどうか判定
        if self.game_state == 0:
            self.game_state = self.board.check_state()
            self.now_player = self.now_player ^ 3
        return True

    def undo_game(self):
        # ゲームを1手戻す（未実装）
        raise Exception("game : undo_game is not implemented")

    def play(self):
        start = time.time()
        try:
            # 1ゲームシミュレートする
            while self.game_state == 0:
                self.process_game()
        except Exception as e:
            print("exception occured in game :")
            print(type(e))
            print(e)
            return "Exception"
        self.record.add_result(self.make_recordresult(self.game_state))
        result = ["processing", f"{self.agent_names[0]}(player 1) win",
                  f"{self.agent_names[1]}(player 2) win", "draw"][self.game_state]
        elapsed_time = time.time() - start
        print(f"{result} {elapsed_time} sec")
        return result

    def play_for_record(self):
        self.play()
        return self.record

    def __init__(self, agent_1, agent_2, first_player=1):
        if first_player < 1 and 2 < first_player:
            raise Exception("first_player must be 1 or 2")
        self.players = [agent_1, agent_2]
        self.agent_names = [agent_1.get_agentname(), agent_2.get_agentname()]
        self.first_player = first_player
        self.board = Board()
        self.prev_move = -1
        self.now_player = first_player
        self.game_state = 0  # 0:ゲーム進行中, 1:player1の勝ち, 2:player2の勝ち, 3:draw
        self.record = Record()   # 棋譜
