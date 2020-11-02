# coding:utf-8

from board import Board


class Game:
    # Ultimate Tic Tac Toeのゲームをシミュレートする
    def process_game(self):
        # ゲームを一手進める
        
        # now_playerに手を聞く
        self.players[self.now_player].request_move()
        # 返ってきた手が有効かどうか調べる

        # 有効じゃない手を打とうとしたら負け

        # ゲーム終了かどうか判定

        # ゲーム終了なら、それぞれのエージェントに結果を返す
        pass
    
    def __init__(self, players):
        self.board = Board()
        self.now_player = 1
        self.players = players