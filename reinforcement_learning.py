# coding:utf-8

import random
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from engine.game_parallel import GameParallel
from learning.learning_util import make_network, make_value_network, convert_board, convert_record, pick_moves, pick_legal_moves, pick_max_legal_moves


# モデルの準備をする（更新するモデルは0とする）
models = [make_network(), make_network()]
value_net = make_value_network()
models[0].load_state_dict(torch.load("./models/alpha_50.pth"))
models[1].load_state_dict(torch.load("./models/alpha_50.pth"))
value_net.load_state_dict(torch.load("./models/value_transfer_fc_3000.pth"))

loss_fn = nn.CrossEntropyLoss(reduction="none")
loss_fn_value = nn.MSELoss()
models[1].eval()

plt_epochs, plt_winrates = [], []

batch_size = 1024
alpha = 0.0000001

# 局面を入れていく
# (局面, 試合のターン数, 報酬=1/-1)
traindata = []

for epoch in range(10000):
    print(f"epoch {epoch} start")

    # 前後入れ替えて1024ゲームずつ同時にやる
    # とりあえず自分の先手だけ
    models[0].eval()
    game_parallel = GameParallel(1024)
    now_player, board_infos = game_parallel.get_nowboards()
    learner_idx = 1
    # learner_idx = 1 if epoch % 2 == 0 else 2
    while len(board_infos) > 0:
        # flat_boardsを変形してTensorにする
        board_data = [convert_board(flat_board, legal_moves)
                    for flat_board, legal_moves in board_infos]
        boards_tensor = torch.Tensor(board_data).cuda()
        outputs = models[0 if now_player == learner_idx else 1](boards_tensor)
        
        moves = pick_legal_moves(
            outputs, [legal for _, legal in board_infos]) if now_player == learner_idx else pick_legal_moves(outputs, [legal for _, legal in board_infos])  # 相手側を確率的なポリシーにする
        # moves = pick_max_legal_moves(
        #     outputs, [legal for _, legal in board_infos]) if now_player == learner_idx else pick_legal_moves(outputs, [legal for _, legal in board_infos])  # 相手側を確率的なポリシーにする
        # moves = pick_legal_moves(
            # outputs, [legal for _, legal in board_infos]) if now_player == learner_idx else pick_max_legal_moves(outputs, [legal for _, legal in board_infos])  # 相手側を決定的なポリシーにする

        game_parallel.process_games(moves)
        
        now_player, board_infos = game_parallel.get_nowboards()

    # 棋譜を取得
    results_and_records_learner = game_parallel.get_results_and_records(learner_idx)
    # results_and_records_opponent = game_parallel.get_results_and_records(learner_idx ^ 3)

    # 棋譜をデータ形式に変換して保管する
    win, lose, draw = 0, 0, 0
    for result, record in results_and_records_learner:
        movenum = len(record)
        reward = 0
        if result == 0:
            win += 1
            reward = 1
            # continue    # 勝ち試合をスキップ
        elif result == 1:
            lose += 1
            reward = -1
            # continue    # 負け試合をスキップ
        else:
            draw += 1
            reward = -1
            # continue
        
        traindata.extend([(convert_record({"board": (board if player_idx == 1 else [[0, 2, 1][mark] for mark in board]), "legal": legal, "move": move}), movenum, reward) for player_idx, _, board, legal, move in record])

    print(f"win={win}, lose={lose}, draw={draw}")
    plt_epochs.append(epoch)
    plt_winrates.append(win / (win + lose + draw))

    # 保管されている局面の数がミニバッチのサイズの50倍を超えたら学習する
    while len(traindata) > batch_size * 50:
        print(len(traindata))
        # ランダムに1バッチぶん取り出す
        random.shuffle(traindata)
        batch_data = traindata[:batch_size]
        traindata = traindata[batch_size:]

        # データを変換する
        data = [data for data, _, _ in batch_data]
        turn_data = [turn for _, turn, _ in batch_data]
        reward_data = [reward for _, _, reward in batch_data]
        board_data = [board for board, _ in data]
        move_data = [move for _, move in data]

        board_tensor = torch.Tensor(board_data).cuda()
        move_tensor = torch.LongTensor(move_data).cuda()

        # 盤面の評価値を推論する
        value_net.eval()
        value_outputs = value_net(board_tensor).cuda()
        value_data = sum(value_outputs.tolist(), [])  # ここ確認
        # print(value_data)

        # lossの係数をつくる
        loss_coef_list = [(reward - value)/turn for turn, reward, value in zip(turn_data, reward_data, value_data)] # ベースラインあり
        # loss_coef_list = [(reward - min(0.9, value - 0.25))/turn for turn, reward, value in zip(turn_data, reward_data, value_data)] # ベースラインあり + 定数
        # loss_coef_list = [(reward - 0.25)/turn for turn, reward, value in zip(turn_data, reward_data, value_data)] # ベースライン定数
        # loss_coef_list = [reward/turn for turn, reward, value in zip(turn_data, reward_data, value_data)] # ベースラインなし
        # loss_coef_list = [reward for turn, reward, value in zip(turn_data, reward_data, value_data)] # 定数
        loss_coef_tensor = torch.Tensor(loss_coef_list).cuda()

        # 学習させる
        models[0].train()
        optimizer = optim.Adagrad(models[0].parameters(), lr=alpha)
        optimizer.zero_grad()
        outputs = models[0](board_tensor).cuda()
        loss = loss_fn(outputs, move_tensor).cuda()
        loss *= loss_coef_tensor
        # print(loss)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

    # 50エポックごとにグラフを出力
    if epoch % 50 == 49:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ln1 = ax1.plot(plt_epochs, plt_winrates, "C0", label="win rate")
        h1, l1 = ax1.get_legend_handles_labels()
        ax1.legend(h1, l1)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("win rate")
        fig.savefig(f"img_{epoch}.png")