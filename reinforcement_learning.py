# coding:utf-8

import concurrent.futures
import random
import torch
import argparse
import matplotlib.pyplot as plt
from torch import nn, optim
from engine.game_parallel import GameParallel
from learning.learning_util import make_policynetwork, make_valuenetwork, convert_board, convert_record, pick_moves, pick_legal_moves, pick_max_legal_moves

parser = argparse.ArgumentParser(description="")
parser.add_argument("--output_path", type=str, help="ポリシーを出力するパス", default="./models/")
parser.add_argument("--batch_size", type=int, help="バッチサイズ", default=8192)
args = parser.parse_args()
batch_size = args.batch_size
output_path = args.output_path

#モデルの準備をする（更新するモデルを0とする）
device_ids = [0, 1]
device = torch.device(f'cuda:{device_ids[0]}')
policy_nets = [make_policynetwork(), make_policynetwork()]
value_net = make_valuenetwork()
policy_nets[0].to(device)
policy_nets[0] = torch.nn.DataParallel(policy_nets[0], device_ids=device_ids)
policy_nets[0].load_state_dict(torch.load("./valuedata_policies/RL_init.pth"))
policy_nets[1].to(device)
policy_nets[1] = torch.nn.DataParallel(policy_nets[1], device_ids=device_ids)
policy_nets[1].load_state_dict(torch.load("./valuedata_policies/RL_init.pth"))
value_net.to(device)
value_net = torch.nn.DataParallel(value_net, device_ids=device_ids)
value_net.load_state_dict(torch.load("./models/value_learn_400.pth"))

loss_fn_policy = nn.CrossEntropyLoss(reduction="none")
loss_fn_value = nn.MSELoss()
policy_nets[1].eval()

plt_epochs, plt_winrates = [], []
alpha = 0.0000001

traindata = []  # (局面、試合のターン数、報酬 = 1/-1)

for epoch in range(10000):
    print(f"epoch {epoch} start")

    # 前後入れ替えてbatch_sizeゲームずつ同時にやる
    policy_nets[0].eval()
    results_and_records_learner = []
    for learner_idx in [1, 2]:
        game_parallel = GameParallel(batch_size)
        now_player, board_infos = game_parallel.get_nowboards()

        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            while len(board_infos) > 0:
                # flat_boardsを変形してTensorにする
                futures = []
                board_data = []
                for flat_board, legal_moves in board_infos:
                    futures.append(executor.submit(convert_board, flat_board, legal_moves))
                for future in futures:
                    board_data.append(future.result())
                boards_tensor = torch.Tensor(board_data).cuda()
                outputs = policy_nets[0 if now_player == learner_idx else 1](boards_tensor)
                moves = pick_legal_moves(
                    outputs, [legal for _, legal in board_infos]) if now_player == learner_idx else pick_legal_moves(outputs, [legal for _, legal in board_infos])  # 相手側を確率的なポリシーにする
                # moves = pick_max_legal_moves(
                #     outputs, [legal for _, legal in board_infos]) if now_player == learner_idx else pick_legal_moves(outputs, [legal for _, legal in board_infos])  # 相手側を確率的なポリシーにする
                # moves = pick_legal_moves(
                #     outputs, [legal for _, legal in board_infos]) if now_player == learner_idx else pick_max_legal_moves(outputs, [legal for _, legal in board_infos])  # 相手側を決定的なポリシーにする
                game_parallel.process_games(moves)
                now_player, board_infos = game_parallel.get_nowboards()

            # 棋譜を取得
            results_and_records_learner += game_parallel.get_results_and_records(learner_idx)

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
            reward = -0.5
            # continue

        traindata.extend([(convert_record({"board": (board if player_idx == 1 else [[0, 2, 1][mark] for mark in board]), "legal": legal, "move": move}), movenum, reward) for player_idx, _, board, legal, move in record])

    print(f"win={win}, lose={lose}, draw={draw}")
    plt_epochs.append(epoch)
    plt_winrates.append(win / (win + lose + draw))

    # 保管されている局面の数がミニバッチのサイズの200倍を超えたら学習する
    while len(traindata) > batch_size * 2:
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
        value_data = sum(value_outputs.tolist(), [])

        # lossの係数をつくる
        # loss_coef_list = [(reward - value)/turn for turn, reward, value in zip(turn_data, reward_data, value_data)] # ベースラインあり
        loss_coef_list = [(reward - value + 0.5)/turn for turn, reward, value in zip(turn_data, reward_data, value_data)] # ベースラインあり + 定数
        # loss_coef_list = [(reward - 0.25)/turn for turn, reward, value in zip(turn_data, reward_data, value_data)] # ベースライン定数
        # loss_coef_list = [reward/turn for turn, reward in zip(turn_data, reward_data)] # ベースラインなし
        # loss_coef_list = [reward for turn, reward, value in zip(turn_data, reward_data, value_data)] # 定数
        loss_coef_tensor = torch.Tensor(loss_coef_list).cuda()

        # 学習させる
        policy_nets[0].train()
        optimizer = optim.Adagrad(policy_nets[0].parameters(), lr=alpha)
        optimizer.zero_grad()
        outputs = policy_nets[0](board_tensor).cuda()
        loss = loss_fn_policy(outputs, move_tensor).cuda()
        loss *= loss_coef_tensor
        loss = loss.mean()
        loss.backward()
        optimizer.step()

    # グラフ出力
    if epoch % 20 == 19:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ln1 = ax1.plot(plt_epochs, plt_winrates, "C0", label="win rate")
        h1, l1 = ax1.get_legend_handles_labels()
        ax1.legend(h1, l1)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("win rate")
        fig.savefig(f"img{device_ids[0]}_{epoch + 1}.png")

        model_path = f"models/test{device_ids[0]}_{epoch + 1}.pth"
        torch.save(policy_nets[0].state_dict(), model_path)