# coding:utf-8

class MiniBoard:
    # 3×3の盤面
    def flatten(self):
        # 現在の盤面を整数のリストにする
        return self.mini_board

    def mark(self, pos, player):
        # idx番目のマス目にplayerの手を打つ
        self.mini_board = player

    def check_state(self):
        # 終了しているかどうかをチェックする
        # 0:続行 1:プレイヤー1の勝ち 2:プレイヤー2の勝ち 3:引き分け
        if self.state != 0:
            return self.state
        bingo = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
                 (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for a, b, c in bingo:
            if self.mini_board[a] != 0 and self.mini_board[a] == self.mini_board[b] == self.mini_board[c]:
                self.state = self.mini_board[a]
                return self.state
        if 0 not in self.mini_board:
            self.state = 3
            return self.state
        return 0

    def __init__(self):
        self.mini_board = [0] * 9
        self.state = 0


class Board:
    # 盤面
    def flatten(self):
        # 現在の盤面を整数のリストにする
        board = sum([mini_board.flatten() for mini_board in  self.mini_boards], [])
        return board

    def legal_moves(self):
        # 合法手のリストを生成する
        legal = [pos for pos, state in enumerate(self.flatten()) if state == 0]
        return legal

    def check_state(self):
        # 終了しているかどうかをチェックする
        # 0:続行 1:プレイヤー1の勝ち 2:プレイヤー2の勝ち 3:引き分け
        return self.big_board.check_state()

    def board_to_miniboard(self, pos):
        # boardのindexを(miniboard番号, miniboardのindex)に変換する
        mini_num, mini_pos = pos / 9, pos % 9
        return (mini_num, mini_pos)

    def mark(self, pos, player):
        # idx番目のマス目にplayerの手を打つ
        mini_num, mini_pos = self.board_to_miniboard(pos)
        self.mini_boards[mini_num].mark(mini_pos, player)
        mini_result = self.mini_boards[mini_num].check_state()
        if mini_result != 0:
            self.big_board.mark(mini_num, player)

    def __init__(self):
        self.mini_boards = [MiniBoard() for _ in range(9)]  # 3×3の小盤面
        self.big_board = MiniBoard()    # 大きな盤面
