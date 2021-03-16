# coding:utf-8

import random
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from engine.game_parallel import GameParallel
from learning.learning_util import make_network, convert_board, convert_record


def pick_moves(outputs):
    # softmaxで手を選択し、Tensorを返す
    softmax = nn.Softmax(dim=1).cuda()
    probs = softmax(outputs)
    moves = torch.multinomial(probs, 1).cuda()
    moves = moves.flatten()
    return moves.tolist()


def pick_legal_moves(outputs, legals):
    # 合法手のなかからsoftmaxで手を選択する
    softmax = nn.Softmax(dim=1).cuda()
    probs = softmax(outputs)
    probs_fixed = torch.Tensor([[prob if idx in legal else 0.0 for idx, prob in enumerate(
        probs)] for probs, legal in zip(probs.tolist(), legals)])
    moves = torch.multinomial(probs_fixed, 1).cuda()
    moves = moves.flatten()
    return moves.tolist()


# モデルの準備をする（更新するモデルは0とする）
models = [make_network(), make_network()]
models[0].load_state_dict(torch.load("./models/test.pth"))
models[1].load_state_dict(torch.load("./models/sota.pth"))
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(models[0].parameters(), lr=0.01)

models[0].eval()
models[1].eval()

batch_size = 1024
alpha = 0.01

# 局面を試合終了までにかかった手数ごとに分けて入れていく
# records_movenum[num][0/1] = num手指して 0:勝ち,1:負け
data_movenum = [[[], []] for _ in range(45)]

for epoch in range(10000):
    print(f"epoch {epoch} start")

    # 前後入れ替えて1024ゲームずつ同時にやる
    models[0].eval()
    gameParallel = GameParallel(1024)
    now_player, board_infos = gameParallel.get_nowboards()
    while len(board_infos) > 0:
        # flat_boardsを変形してTensorにする
        board_data = [convert_board(flat_board, legal_moves)
                    for flat_board, legal_moves in board_infos]
        boards_tensor = torch.Tensor(board_data).cuda()
        outputs = models[now_player - 1](boards_tensor)
        
        moves = pick_legal_moves(
            outputs, [legal for _, legal in board_infos]) if now_player == 1 else pick_legal_moves(outputs, [legal for _, legal in board_infos])
        gameParallel.process_games(moves)
        
        now_player, board_infos = gameParallel.get_nowboards()

    # 棋譜をデータ形式に変換して、指した手数ごとにわけて保管する
    results_and_records = gameParallel.get_results_and_records(1)
    win, lose = 0, 0
    for result, record in results_and_records:
        movenum = len(record)
        if result == 0:
            win += 1
        else:
            lose += 1
        data_movenum[movenum][result].extend([convert_record({"board": (board if player_idx == 1 else [[0, 2, 1][mark] for mark in board]), "legal": legal, "move": move}) for player_idx, _, board, legal, move in record])

    print(f"win={win}, lose={lose}")

    # 保管されている棋譜がミニバッチのサイズの2倍を越えたら学習する
    models[0].train()
    for movenum, data_list in enumerate(data_movenum):
        for result, data in enumerate(data_list):
            while len(data) >= batch_size * 2:
                if result == 1:
                    # いったん勝ち試合のみで
                    data = []
                    break

                # print(f"learn: {movenum, result}")
                # ランダムに1バッチぶん取り出す
                random.shuffle(data)
                batch_data = data[::batch_size]
                data = data[batch_size::]

                # データを変換する
                board_data, move_data = [board for board, _ in data], [move for _, move in data]
                board_tensor = torch.Tensor(board_data).cuda()
                move_tensor = torch.LongTensor(move_data).cuda()

                # 学習率を変更
                beta = alpha * (1.0 if result == 0 else -1.0) / movenum
                optimizer = optim.Adagrad(models[0].parameters(), lr=beta)

                # 学習させる
                optimizer.zero_grad()
                outputs = models[0](board_tensor).cuda()
                loss = loss_fn(outputs, move_tensor).cuda()
                loss.backward()
                optimizer.step()

        