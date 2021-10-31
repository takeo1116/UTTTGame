# coding:utf-8

import multiprocessing
import os
import sys
import torch
import argparse
import subprocess
import concurrent.futures
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn.modules.activation import LeakyReLU
from torch.utils.data import TensorDataset, DataLoader
from learn.network.network import make_policynetwork, make_valuenetwork
from learn.util.feature import convert_movedata, make_feature
from learn.util.reinforcementtrainer import ReinfocementTrainer
from learn.util.recordreader import RecordReader
from engine.parallelgame import ParallelGames
from learn.util.pickmove import pick_legalmoves


parser = argparse.ArgumentParser(description="自己対戦による強化学習")
parser.add_argument("--gpu_ids", type=str, default=0, help="使用するgpuのid(カンマ区切り)")
parser.add_argument("--temperature", type=float,
                    default=1.0, help="modelが手を選択するときの温度")
parser.add_argument("--policy_batchsize", type=int,
                    default=256, help="一度に並列対戦する回数（policy）")
parser.add_argument("--policy_batchnum", type=int, default=250,
                    help="1epochあたりに何batch対戦するか（policy）")
parser.add_argument("--value_batchsize", type=int,
                    default=8192, help="一度に並列対戦する回数（value）")
parser.add_argument("--value_batchnum", type=int, default=35,
                    help="1epochあたりに何batch対戦するか（value）")
parser.add_argument("--output_path", type=str,
                    default="./learn/RL_output", help="グラフ、モデルをoutputするディレクトリ")
parser.add_argument("--init_policy", type=str,
                    default=None, help="強化学習の初期ポリシー")
parser.add_argument("--init_value", type=str, default=None, help="初期value")
parser.add_argument("--policy_lr", type=float,
                    default=0.0000001, help="policyの基本学習率")
parser.add_argument("--value_lr", type=float,
                    default=0.000001, help="valueの基本学習率")
args = parser.parse_args()

# outputするディレクトリの生成
os.makedirs(args.output_path, exist_ok=True)
models_path = os.path.join(args.output_path, "models")
values_path = os.path.join(args.output_path, "values")
graphs_path = os.path.join(args.output_path, "graphs")
policy_movedata_path = os.path.join(args.output_path, "policy_movedata")
value_movedata_path = os.path.join(args.output_path, "value_movedata")
os.makedirs(models_path, exist_ok=True)
os.makedirs(values_path, exist_ok=True)
os.makedirs(graphs_path, exist_ok=True)
os.makedirs(policy_movedata_path, exist_ok=True)
os.makedirs(value_movedata_path, exist_ok=True)
# os.makedirs(os.path.join(args.output_path, "log"), exist_ok=True)

# モデルのロード
gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
no0_ids = [gpu_id for gpu_id in gpu_ids if gpu_id > 0]
cpu_num = min(16, multiprocessing.cpu_count())
print(f"gpu_ids = {gpu_ids}")
learner = make_policynetwork()
learner.cuda()
learner = torch.nn.DataParallel(learner, device_ids=gpu_ids)
learner.load_state_dict(torch.load(args.init_policy), strict=False)
torch.save(learner.state_dict(), f"{args.output_path}/models/policy_{0}.pth")

sl = make_policynetwork()
sl.cuda()
sl = torch.nn.DataParallel(sl, device_ids=gpu_ids)
sl.load_state_dict(torch.load(args.init_policy), strict=False)

value = make_valuenetwork()
value.cuda()
value = torch.nn.DataParallel(value, device_ids=gpu_ids)
value.load_state_dict(torch.load(args.init_value), strict=False)

plt_epochs, plt_winrates = [], []

def play_parallelgames(parallel_num, first_model, second_model, temp=1.0, first_name="first", second_name="second"):
    # parallel_num並列対戦をしてgamesを返す
    first_model.eval()
    second_model.eval()
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_num) as executor:
        games = ParallelGames(parallel_num)
        player_idx, processing_boards = games.get_nowboards()
        while len(processing_boards) > 0:
            model, agent_name = (first_model, first_name) if player_idx == 1 else (
                second_model, second_name)
            flatboardlist = [
                flat_board for flat_board, _ in processing_boards]
            legalmoveslist = [legal_moves for _,
                                legal_moves in processing_boards]
            results = executor.map(make_feature, flatboardlist, legalmoveslist, chunksize=max(
                1024, len(processing_boards)//cpu_num))
            features = [feature for feature in results]
            feature_tensor = torch.Tensor(features).cuda()
            outputs = model(feature_tensor)
            moves = pick_legalmoves(
                outputs, [legal_moves for _, legal_moves in processing_boards], temp)
            games.process_games(moves, agent_name)

            player_idx, processing_boards = games.get_nowboards()
        games.add_recordresults()
    return games

for epoch in range(10000):
    print(f"epoch {epoch} start")

    # policyの学習
    print("make policymovedata")
    procs = []
    for gpu_idx in no0_ids:
        proc = subprocess.Popen(["python3", "-m", "learn.reinforcement.make_policymovedata", "--process_id", str(gpu_idx), "--gpu_ids", str(gpu_idx), "--policy_batchsize", str(args.policy_batchsize), "--policy_batchnum", str(args.policy_batchnum//len(no0_ids)), "--output_path", policy_movedata_path, "--enemy_policy_path", models_path, "--learner_policy", f"{args.output_path}/models/policy_{epoch}.pth"], stdout=sys.stdout, stderr=sys.stdout)
        procs.append(proc)
    [proc.wait() for proc in procs]

    print("train policymovedata")
    policy_recordreader = RecordReader(policy_movedata_path)
    movedatalist = policy_recordreader.get_movedatalist()

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_num) as executor:
        results = executor.map(
            convert_movedata, movedatalist, chunksize=len(movedatalist)//cpu_num)
        result_list = [result for result in results]
        feature_tensor = torch.Tensor(
            [feature for feature, _, _ in result_list]).cuda()
        move_tensor = torch.LongTensor([move for _, move, _ in result_list]).cuda()
        rewards_tensor = torch.Tensor(
            [reward for _, _, reward in result_list]).cuda()
        dataset = TensorDataset(
            feature_tensor, move_tensor, rewards_tensor)
        policy_dataloader = DataLoader(dataset, 8192, shuffle=True)
    
        # policyを1エポック学習する
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        optimizer = optim.Adagrad(learner.parameters(), lr=args.policy_lr)
        value.eval()
        learner.train()
        for feature_data, move_data, reward_data in policy_dataloader:
            # 推論
            optimizer.zero_grad()
            outputs = learner(feature_data).cuda()

            # valueを推論する
            value_outputs = value(feature_data).cuda()
            value_data = sum(value_outputs.tolist(), [])
            coef_tensor = torch.Tensor(
                [(reward - val) for reward, val in zip(sum(reward_data.tolist(), []), value_data)]).cuda()

            loss = loss_fn(outputs, move_data).cuda()
            loss *= coef_tensor
            loss = loss.mean()
            loss.backward()
            optimizer.step()

    torch.save(learner.state_dict(), f"{args.output_path}/models/policy_{epoch + 1}.pth")

    # valueの学習
    print("make valuemovedata")
    procs = []
    for gpu_idx in no0_ids:
        proc = subprocess.Popen(["python3", "-m", "learn.reinforcement.make_valuemovedata", "--process_id", str(gpu_idx), "--gpu_ids", str(gpu_idx), "--value_batchsize", str(args.value_batchsize), "--value_batchnum", str(args.value_batchnum//len(no0_ids)), "--output_path", value_movedata_path, "--learner_policy", f"{args.output_path}/models/policy_{epoch + 1}.pth", "--sl_policy", args.init_policy], stdout=sys.stdout, stderr=sys.stdout)
        procs.append(proc)
    [proc.wait() for proc in procs]

    # valueを1エポック学習する
    print("train valuedata")
    value_recordreader = RecordReader(value_movedata_path)
    movedatalist = policy_recordreader.get_movedatalist()
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_num) as executor:
        results = executor.map(
            convert_movedata, movedatalist, chunksize=len(movedatalist)//cpu_num)
        result_list = [result for result in results]
        feature_tensor = torch.Tensor(
            [feature for feature, _, _ in result_list]).cuda()
        value_tensor = torch.Tensor(
            [value for _, _, value in result_list]).cuda()
        dataset = TensorDataset(
            feature_tensor, value_tensor)
        value_dataloader = DataLoader(dataset, 8192, shuffle=True)

        value.train()
        loss_fn = nn.MSELoss(reduction="none")
        optimizer = optim.SGD(value.parameters(), lr=args.value_lr)
        for feature_tensor, value_tensor in value_dataloader:
            optimizer.zero_grad()
            outputs = value(feature_tensor).cuda()
            loss = loss_fn(outputs, value_tensor).cuda()
            loss = loss.mean()
            loss.backward()
            optimizer.step()
    
    # 強さチェック
    print("check winrate")
    learner.eval()
    sl.eval()
    learner_win, sl_win = 0, 0
    for idx, models in enumerate([(learner, sl, "learner", "slpolicy"), (sl, learner, "slpolicy", "learner")]):
        learner_idx = idx + 1
        first_model, second_model, first_name, second_name = models
        games = play_parallelgames(
            10000 // 2, first_model, second_model, args.temperature, first_name, second_name)
        learner_win += sum([1 if state ==
                            learner_idx else 0 for state in games.game_states])
        sl_win += sum([1 if state == learner_idx ^
                        3 else 0 for state in games.game_states])
    print(f"check winrate : win={learner_win}, lose={sl_win}, draw={10000 - (learner_win + sl_win)}")

    plt_epochs.append(epoch)
    plt_winrates.append(learner_win / 10000)

    if epoch % 10 == 9:
        # valueのセーブ
        value_path = f"{values_path}/value_{epoch + 1}.pth"
        torch.save(value.state_dict(), value_path)

        # グラフ出力
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ln1 = ax1.plot(plt_epochs, plt_winrates, "C0", label="win rate")
        h1, l1 = ax1.get_legend_handles_labels()
        ax1.legend(h1, l1)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("win rate")
        fig.savefig(f"{args.output_path}/graphs/img_{epoch + 1}.png")