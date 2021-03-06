# coding:utf-8

import os
import json


def multiply_dataset(datas):
    # データを反転、回転して増やす
    # ゲームエンジンと盤面のindexの変換（逆変換も同じ形）
    engine_to_board = [0, 1, 2, 9, 10, 11, 18, 19, 20, 3, 4, 5, 12, 13, 14, 21, 22, 23, 6, 7, 8, 15, 16, 17, 24, 25, 26, 27, 28, 29, 36, 37, 38, 45, 46, 47, 30, 31, 32, 39, 40, 41, 48, 49, 50, 33, 34, 35, 42, 43, 44, 51, 52, 53, 54, 55, 56, 63, 64, 65, 72, 73, 74, 57, 58, 59, 66, 67, 68, 75, 76, 77, 60, 61, 62, 69, 70, 71, 78, 79, 80]
    perms = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
        [72, 63, 54, 45, 36, 27, 18, 9, 0, 73, 64, 55, 46, 37, 28, 19, 10, 1, 74, 65, 56, 47, 38, 29, 20, 11, 2, 75, 66, 57, 48, 39, 30, 21, 12, 3, 76, 67, 58, 49, 40, 31, 22, 13, 4, 77, 68, 59, 50, 41, 32, 23, 14, 5, 78, 69, 60, 51, 42, 33, 24, 15, 6, 79, 70, 61, 52, 43, 34, 25, 16, 7, 80, 71, 62, 53, 44, 35, 26, 17, 8],
        [80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [8, 17, 26, 35, 44, 53, 62, 71, 80, 7, 16, 25, 34, 43, 52, 61, 70, 79, 6, 15, 24, 33, 42, 51, 60, 69, 78, 5, 14, 23, 32, 41, 50, 59, 68, 77, 4, 13, 22, 31, 40, 49, 58, 67, 76, 3, 12, 21, 30, 39, 48, 57, 66, 75, 2, 11, 20, 29, 38, 47, 56, 65, 74, 1, 10, 19, 28, 37, 46, 55, 64, 73, 0, 9, 18, 27, 36, 45, 54, 63, 72],
        [72, 73, 74, 75, 76, 77, 78, 79, 80, 63, 64, 65, 66, 67, 68, 69, 70, 71, 54, 55, 56, 57, 58, 59, 60, 61, 62, 45, 46, 47, 48, 49, 50, 51, 52, 53, 36, 37, 38, 39, 40, 41, 42, 43, 44, 27, 28, 29, 30, 31, 32, 33, 34, 35, 18, 19, 20, 21, 22, 23, 24, 25, 26, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [80, 71, 62, 53, 44, 35, 26, 17, 8, 79, 70, 61, 52, 43, 34, 25, 16, 7, 78, 69, 60, 51, 42, 33, 24, 15, 6, 77, 68, 59, 50, 41, 32, 23, 14, 5, 76, 67, 58, 49, 40, 31, 22, 13, 4, 75, 66, 57, 48, 39, 30, 21, 12, 3, 74, 65, 56, 47, 38, 29, 20, 11, 2, 73, 64, 55, 46, 37, 28, 19, 10, 1, 72, 63, 54, 45, 36, 27, 18, 9, 0],
        [8, 7, 6, 5, 4, 3, 2, 1, 0, 17, 16, 15, 14, 13, 12, 11, 10, 9, 26, 25, 24, 23, 22, 21, 20, 19, 18, 35, 34, 33, 32, 31, 30, 29, 28, 27, 44, 43, 42, 41, 40, 39, 38, 37, 36, 53, 52, 51, 50, 49, 48, 47, 46, 45, 62, 61, 60, 59, 58, 57, 56, 55, 54, 71, 70, 69, 68, 67, 66, 65, 64, 63, 80, 79, 78, 77, 76, 75, 74, 73, 72],
        [0, 9, 18, 27, 36, 45, 54, 63, 72, 1, 10, 19, 28, 37, 46, 55, 64, 73, 2, 11, 20, 29, 38, 47, 56, 65, 74, 3, 12, 21, 30, 39, 48, 57, 66, 75, 4, 13, 22, 31, 40, 49, 58, 67, 76, 5, 14, 23, 32, 41, 50, 59, 68, 77, 6, 15, 24, 33, 42, 51, 60, 69, 78, 7, 16, 25, 34, 43, 52, 61, 70, 79, 8, 17, 26, 35, 44, 53, 62, 71, 80]
    ]

    new_datas = []
    for boards, move in datas:
        for perm in perms:
            new_boards = []
            for board in boards:
                flatten = sum(board, [])
                new_flatten = [flatten[perm[idx]] for idx in range(81)]
                new_boards.append([new_flatten[i * 9: i * 9 + 9] for i in range(9)])
            new_move = engine_to_board[perm.index(engine_to_board[move])]
            new_datas.append((new_boards, new_move))
    
    return new_datas


# datasetを読み込んで、回転・反転によって8倍に増やす
path = "datasets/MctsAgent_10000"
with open(f"{path}/train.json") as f:
    original_traindata = json.load(f)

# n回に分けてセーブする
n = 10
data_num = len(original_traindata)
print(f"data_num = {data_num}")
for i in range(n):
    # if i != 0:
    #     continue
    print(i * (data_num // n), (i + 1) * (data_num // n))
    part = original_traindata[i * (data_num // n): (i + 1) * (data_num // n)]
    new_dataset = multiply_dataset(part)
    
    with open(f"{path}/part_{i}.json", mode="w") as f:
        json.dump(new_dataset, f)

