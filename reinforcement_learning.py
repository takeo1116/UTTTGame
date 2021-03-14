# coding:utf-8

import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from engine.game_parallel import GameParallel
from learning.learning_util import make_network, convert_board


def pick_moves(outputs):
    # softmaxで手を選択し、Tensorを返す
    softmax = nn.Softmax(dim=1).cuda()
    probs = softmax(outputs)
    moves = torch.multinomial(probs, 1).cuda()
    moves = moves.flatten()
    return moves.tolist()


def pick_legal_moves(outputs, legals):
    # 合法手のなかからsoftmaxで手を選択する
    probs_list = outputs.tolist()
    probs_list_fixed = [[prob if idx in legal else 0.0 for idx, prob in enumerate(
        probs)] for probs, legal in zip(probs_list, legals)]
    outputs_fixed = torch.Tensor(probs_list_fixed).cuda()
    return pick_moves(outputs_fixed)


# モデルの準備をする（更新するモデルは0とする）
models = [make_network(), make_network()]
models[0].load_state_dict(torch.load("./models/test.pth"))
models[1].load_state_dict(torch.load("./models/sota.pth"))
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(models[0].parameters(), lr=0.01)

models[0].eval()
models[1].eval()

# 前後入れ替えて1024ゲームずつ同時にやる
gameParallel = GameParallel(1024)
now_player, board_infos = gameParallel.get_nowboards()
print("start")
while len(board_infos) > 0:
    # flat_boardsを変形してTensorにする
    # print(board_infos)
    board_data = [convert_board(flat_board, legal_moves)
                  for flat_board, legal_moves in board_infos]
    boards_tensor = torch.Tensor(board_data).cuda()
    outputs = models[now_player - 1](boards_tensor)
    
    moves = pick_moves(
        outputs) if now_player == 1 else pick_legal_moves(outputs, [legal for _, legal in board_infos])
    gameParallel.process_games(moves)
    
    now_player, board_infos = gameParallel.get_nowboards()
print("end")