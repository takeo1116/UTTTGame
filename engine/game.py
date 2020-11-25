# coding:utf-8

from .board import Board
from agent.random_agent import RandomAgent
from agent.mcts_agent import MctsAgent
import time


class Game:
    # Ultimate Tic Tac Toeのゲームをシミュレートする
    def process_game(self):
        # ゲームを一手進める
        flat_board = self.board.flatten()
        prev_move = -1
        if len(self.game_record) > 0:
            _, _, _, prev_move = self.game_record[-1]
        legal_moves = self.board.legal_moves(prev_move)
        # now_playerに手を聞く
        move = self.players[self.now_player - 1].request_move(
            flat_board.copy(), legal_moves.copy(), self.now_player)
        # 返ってきた手が有効かどうか調べる（有効じゃない手を打とうとしたら負け）
        if move not in legal_moves:
            self.game_state = self.now_player ^ 3
        # 手を反映させる
        self.board.mark(move, self.now_player)
        self.game_record.append((flat_board, legal_moves, self.now_player, move))
        # ゲーム終了かどうか判定
        if self.game_state == 0:
            self.game_state = self.board.check_state()
            self.now_player = self.now_player ^ 3
        return True

    def undo_game(self):
        # ゲームを一手戻す
        # 戻せなかった場合Falseを返す
        if len(self.game_record) == 0:
            return False
        _, _, prev_player, prev_pos = self.game_record.pop(-1)
        self.board.unmark(prev_pos)
        self.now_player = prev_player
        return True

    def play(self):
        start = time.time()
        # 1ゲームシミュレートする
        while self.game_state == 0:
            self.process_game()
        # ゲーム終了したら、それぞれのエージェントに結果を返す
        flat_board = self.board.flatten()
        self.players[0].game_end(flat_board, 1, self.game_state)
        self.players[1].game_end(flat_board, 2, self.game_state)
        result = ["processing", "player 1 win",
                  "player 2 win", "draw"][self.game_state]
        elapsed_time = time.time() - start
        print(f"{result} {elapsed_time} sec")
        return result

    def constract_agent(self, agent_name):
        options = agent_name.split("_")
        # エージェントの名前から新品のインスタンスを返す
        if options[0] == "RandomAgent":
            return RandomAgent()
        elif options[0] == "MctsAgent":
            playout_num = int(options[1])
            return MctsAgent(playout_num)
        else:
            return None

    def print_board(self):
        # コンソールに盤面を出力する
        nums = [0, 1, 2, 9, 10, 11, 18, 19, 20]
        flat_board = self.board.flatten()
        for row in range(9):
            string = ""
            for idx, num in enumerate(nums):
                string += str(flat_board[num + 3 * row + row // 3 * 18])
                if idx in [2, 5]:
                    string += " "
            print(string)
            if row in [2, 5]:
                print()

    def gat_record(self):
        # 棋譜を取得する
        return self.game_record

    def __init__(self, agent_name_1, agent_name_2):
        # players = [agent1, agent2]
        self.board = Board()
        self.now_player = 1
        self.players = [self.constract_agent(
            agent_name_1), self.constract_agent(agent_name_2)]
        self.game_state = 0  # 0:ゲーム進行中, 1:player1の勝ち, 2:player2の勝ち
        self.game_record = []   # 棋譜[(着手前のflat_board, legal, 着手したplayer, move)]
