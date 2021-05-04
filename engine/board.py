# coding:utf-8

class LocalBoard:
    # 3×3の盤面
    def flatten(self):
        # 現在の盤面を整数のリストにする
        return self.local_board

    def mark(self, pos, player):
        # idx番目のマス目にplayerの手を打つ
        self.local_board[pos] = player

    def check_state(self):
        # 終了しているかどうかをチェックする
        # 0:続行 1:プレイヤー1の勝ち 2:プレイヤー2の勝ち 3:引き分け
        if self.state != 0:
            return self.state
        bingo = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
                 (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for a, b, c in bingo:
            if self.local_board[a] != 0 and self.local_board[a] == self.local_board[b] == self.local_board[c]:
                self.state = self.local_board[a]
                return self.state
        if 0 not in self.local_board:
            self.state = 3
            return self.state
        return 0

    def __init__(self, flat_board=None):
        # flat_boardに初期状態を入力可能
        self.local_board = [0] * 9 if flat_board is None else flat_board
        self.state = 0
        self.state = self.check_state()


class Board:
    # 盤面
    def flatten(self):
        # 現在の盤面を整数のリストにする
        board = sum([local_board.flatten()
                     for local_board in self.local_boards], [])
        return board

    def legal_moves(self, prev_move):
        # 合法手のリストを生成する
        legal_area = prev_move % 9
        legal = []
        if prev_move == -1 or self.grobal_board.flatten()[legal_area] != 0:
            # legal = [pos for pos, state in enumerate(
            #     self.flatten()) if state == 0]
            legal = [pos for pos, state in enumerate(
                self.flatten()) if state == 0 and self.grobal_board.flatten()[pos // 9] == 0]
        else:
            legal = [legal_area * 9 + pos for pos,
                     state in enumerate(self.flatten()[legal_area * 9: legal_area * 9 + 9]) if state == 0]
        return legal

    def check_state(self):
        # 終了しているかどうかをチェックする
        # 0:続行 1:プレイヤー1の勝ち 2:プレイヤー2の勝ち 3:引き分け
        return self.grobal_board.check_state()

    def board_to_localboard(self, pos):
        # boardのindexを(localboard番号, localboardのindex)に変換する
        local_num, local_pos = pos // 9, pos % 9
        return (local_num, local_pos)

    def mark(self, pos, player):
        # idx番目のマス目にplayerの手を打つ
        local_num, local_pos = self.board_to_localboard(pos)
        self.local_boards[local_num].mark(local_pos, player)
        local_result = self.local_boards[local_num].check_state()
        self.grobal_board.mark(local_num, local_result)

    def unmark(self, pos):
        # idx番目のマス目に書いてあるマークを消す
        self.mark(pos, 0)

    def __init__(self, flat_board=None):
        # flat_boardに初期状態を入力可能
        self.local_boards = [LocalBoard() for _ in range(9)] if flat_board is None else [LocalBoard(
            flat_board[i * 9: (i + 1) * 9]) for i in range(9)]  # 3x3の小盤面
        self.grobal_board = LocalBoard(
            [local.check_state() for local in self.local_boards])    # 大きな盤面
