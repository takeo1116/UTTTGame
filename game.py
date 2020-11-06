# coding:utf-8

from board import Board
from random_agent import RandomAgent

class Game:
    # Ultimate Tic Tac Toeのゲームをシミュレートする
    def process_game(self):
        # ゲームを一手進める
        flat_board = self.board.flatten()
        legal_moves = self.board.legal_moves()
        # now_playerに手を聞く
        move = self.players[self.now_player - 1].request_move(
            flat_board.copy(), legal_moves.copy(), self.now_player)
        # 返ってきた手が有効かどうか調べる（有効じゃない手を打とうとしたら負け）
        if move not in legal_moves:
            self.game_state = self.now_player ^ 3
        # 手を反映させる
        self.board.mark(move, self.now_player)
        # ゲーム終了かどうか判定
        if self.game_state == 0:
            self.game_state = self.board.check_state()
            self.now_player = self.now_player ^ 3

    def play(self):
        # 1ゲームシミュレートする（いい関数名？）
        while self.game_state == 0:
            self.process_game()
            print(self.now_player)
        # ゲーム終了したら、それぞれのエージェントに結果を返す
        flat_board = self.board.flatten()
        self.players[0].game_end(flat_board, 1, self.game_state)
        self.players[1].game_end(flat_board, 2, self.game_state)
        self.print_board()

    def constract_agent(self, agent_name):
        # エージェントの名前から新品のインスタンスを返す
        if agent_name == "RandomAgent":
            return RandomAgent()
        else:
            return None

    def print_board(self):
        # コンソールに盤面を出力する
        flat_board = self.board.flatten()
        for i in range(0, 81, 9):
            row = ""
            for j in range(9):
                row += str(flat_board[i + j])
                if j in [2, 5]:
                    row += " "
            print(row)
            if i in [18, 45]:
                print()


    def __init__(self, agent_name_1, agent_name_2):
        # players = [agent1, agent2]
        self.board = Board()
        self.now_player = 1
        self.players = [self.constract_agent(agent_name_1), self.constract_agent(agent_name_2)]
        self.game_state = 0  # 0:ゲーム進行中, 1:player1の勝ち, 2:player2の勝ち
