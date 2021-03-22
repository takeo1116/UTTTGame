# coding:utf-8

import torch
from torch import nn, optim
from .network import Network, ValueNetwork


def convert_board(board, legal):
    # boardとlegalを学習、推論に適した形に変換する
    # 9*9の盤面で、自分のマーク、相手のマーク、着手可能な場所の3チャネル
    bingo = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
                     (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    def paint_won_board(board):
        # すでに取られたlocalboardを塗りつぶしたものを返す
        def paint_won_localboard(localboard):
            for a, b, c in bingo:
                if localboard[a] == localboard[b] == localboard[c] == 1:
                    return [1 for _ in range(9)]
                if localboard[a] == localboard[b] == localboard[c] == 2:
                    return [2 for _ in range(9)]
            return localboard

        painted_board = [paint_won_localboard(
            board[local_idx * 9 : local_idx * 9 + 9]) for local_idx in range(9)]
        return sum(painted_board, [])

    def make_chanceboard(board, player_num):
        # player_numが置くとローカルボードを取れる場所を塗りつぶしたもの
        def is_won(localboard):
            # そのlocal_boardが取られているか
            for a, b, c in bingo:
                if localboard[a] == localboard[b] == localboard[c] and localboard[a] != 0:
                    return True
            return False

        def check_place(idx):
            # 指定された場所に置くとローカルボードを取れるか？
            local_idx = idx // 9
            localboard = board[local_idx * 9 : local_idx * 9 + 9]
            if board[idx] != 0 or is_won(localboard):
                return False
            localboard[idx % 9] = player_num
            return is_won(localboard)

        chance_board = [1 if check_place(idx) else 0 for idx in range(81)]
        return chance_board

    def reshape(board):
        # boardの81要素を並べ変えて9*9の盤面（実際の盤面と同じ並び）を作る
        row = [0, 1, 2, 9, 10, 11, 18, 19, 20]
        column = [0, 3, 6, 27, 30, 33, 54, 57, 60]
        return [[board[r + c] for r in row] for c in column]
        
    board = paint_won_board(board)
    my_board = [1 if mark == 1 else 0 for mark in board]
    op_board = [1 if mark == 2 else 0 for mark in board]
    legal_board = [1 if pos in legal else 0 for pos in range(81)]
    my_chanceboard = make_chanceboard(board, 1)
    op_chanceboard = make_chanceboard(board, 2)
    return [reshape(brd) for brd in [my_board, op_board, legal_board, my_chanceboard, op_chanceboard]]


def convert_record(record):
    # recordを加工して、学習に適した教師データに変換する
    board, legal, move = record["board"], record["legal"], record["move"]
    board_data = convert_board(board, legal)
    return (board_data, move)


def make_network():
    # モデルを作る
    channels_num = 5
    model = Network(channels_num)
    return model


def make_value_network():
    # バリューネットワークを作る
    channels_num = 5
    model = ValueNetwork(channels_num)
    return model
    