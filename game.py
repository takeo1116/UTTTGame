# coding:utf-8

from board import Board


class Game:
    # Ultimate Tic Tac Toeのゲームをシミュレートする
    def process_game(self):
        # ゲームを一手進める
        flat_board = self.board.flatten()
        legal_moves = self.board.legal_moves()
        
        # now_playerに手を聞く
        move = self.players[self.now_player].request_move(flat_board.copy(), legal_moves.copy(), self.now_player)
        # 返ってきた手が有効かどうか調べる（有効じゃない手を打とうとしたら負け）
        if move not in legal_moves:
            self.game_state = 1 - self.now_player
        # 手を反映させる
        self.board.mark(move, self.now_player)
        # ゲーム終了かどうか判定
        if self.game_state == 0:
            self.game_state = self.board.check_state()
        # ゲーム終了なら、それぞれのエージェントに結果を返す
        if self.game_state != 0:
            
        # player番号を次に進める
        self.now_player = 1 - self.now_player
        pass
    
    def __init__(self, players):
        self.board = Board()
        self.now_player = 1
        self.players = players
        self.game_state = 0 # 0:ゲーム進行中, 1:player1の勝ち, 2:player2の勝ち