# coding:utf-8

import torch
from torch.utils.data import TensorDataset, DataLoader
from engine.record import MoveDataResult


def make_feature(movedata):
    # movedataから特徴量の形に変換する
    # 9*9の盤面で、自分のマーク、相手のマーク、着手可能な場所の3チャネル
    bingo = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
             (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    def paint_won_board(flat_board):
        # すでに取られたlocalboardを塗りつぶしたものを返す
        def paint_won_localboard(localboard):
            for a, b, c in bingo:
                if localboard[a] == localboard[b] == localboard[c] == 1:
                    return [1 for _ in range(9)]
                if localboard[a] == localboard[b] == localboard[c] == 2:
                    return [2 for _ in range(9)]
            return [mark for mark in localboard]    # そのまま

        painted_board = [paint_won_localboard(
            flat_board[local_idx * 9: local_idx * 9 + 9]) for local_idx in range(9)]
        return sum(painted_board, [])

    def make_chanceboard(flat_board, player_num):
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
            localboard = flat_board[local_idx * 9: local_idx * 9 + 9]
            if flat_board[idx] != 0 or is_won(localboard):
                return False
            localboard[idx % 9] = player_num
            return is_won(localboard)

        chance_board = [1 if check_place(idx) else 0 for idx in range(81)]
        return chance_board

    def to_model_index(flat_board):
        # engineのindexからmodelのindexに変換する
        row = [0, 1, 2, 9, 10, 11, 18, 19, 20]
        column = [0, 3, 6, 27, 30, 33, 54, 57, 60]
        return [[flat_board[r + c] for r in row] for c in column]

    painted = paint_won_board(movedata.flat_board)
    my_board = [1 if mark == 1 else 0 for mark in painted]
    op_board = [1 if mark == 2 else 0 for mark in painted]
    legal_board = [
        1 if pos in movedata.legal_moves else 0 for pos in range(81)]
    my_chanceboard = make_chanceboard(painted, 1)
    op_chanceboard = make_chanceboard(painted, 2)

    return [to_model_index(brd) for brd in [my_board, op_board, legal_board, my_chanceboard, op_chanceboard]]


def make_dataset(movedatalist):
    def get_value(movedata):
        result = movedata.result
        if result == MoveDataResult.WIN:
            return 1
        elif result == MoveDataResult.LOSE:
            return -1
        elif result == MoveDataResult.DRAW:
            return -0.5
        else:
            return 0

    # movedatalistから教師データ(feature, move, value)を作る
    features = []
    moves = []
    values = []
    for movedata in movedatalist:
        features.append(make_feature(movedata))
        moves.append(movedata.move)
        values.append(get_value(movedata.result))

    feature_tensor = torch.Tensor(features).cuda()
    move_tensor = torch.LongTensor(moves).cuda()
    value_tensor = torch.Tensor(values).cuda()

    dataset = TensorDataset(feature_tensor, move_tensor, value_tensor)
    return dataset
