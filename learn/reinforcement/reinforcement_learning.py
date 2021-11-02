# coding:utf-8

import multiprocessing
import os
import sys
import torch
import json
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
parser.add_argument("--policy_parallelnum", type=int, default=4,
                    help="policy_movedataの生成時に1GPUで何プロセス同時に起動するか")                  
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

#output_pathにsavedata.jsonがあるとき、前回の続きから始める

epoch_offset = 0
savedata_path = os.path.join(args.output_path, f"savedata.json")
if os.path.isfile(savedata_path):
    with open(savedata_path) as f:
        savedata = json.load(f)
        epoch_offset = savedata["len"]

print(f"start with epoch {epoch_offset}")

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

gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]

for epoch in range(epoch_offset, 10000):
    print(f"epoch {epoch} start")

    # pathの管理
    learner_path = os.path.join(models_path, f"policy_{epoch}.pth")
    sl_path = os.path.join(models_path, f"policy_{0}.pth")
    value_path = os.path.join(values_path, f"value_{epoch}.pth")
    new_learner_path = os.path.join(models_path, f"policy_{epoch + 1}.pth")
    new_value_path = os.path.join(values_path, f"value_{epoch + 1}.pth")

    # policyの学習
    print("make policymovedata")
    procs = []
    for parallel_idx in range(args.policy_parallelnum):
        for gpu_idx in gpu_ids:
            proc = subprocess.Popen(["python3", "-m", "learn.reinforcement.make_policymovedata", "--process_id", str(parallel_idx * (max(gpu_ids) + 1) + gpu_idx), "--gpu_ids", str(gpu_idx), "--policy_batchsize", str(args.policy_batchsize), "--policy_batchnum", str(args.policy_batchnum//(len(gpu_ids) * args.policy_parallelnum)), "--output_path", policy_movedata_path, "--enemy_policy_path", models_path, "--learner_policy", learner_path, "--temperature", str(args.temperature)], stdout=sys.stdout, stderr=sys.stdout)
            procs.append(proc)
    [proc.wait() for proc in procs]

    # print("train policynetwork")
    # proc = subprocess.Popen(["python3", "-m", "learn.reinforcement.train_policynetwork", "--movedata_path", policy_movedata_path, "--learner_path", learner_path, "--value_path", value_path, "--output_path", new_learner_path, "--policy_lr", str(args.policy_lr)], stdout=sys.stdout, stderr=sys.stdout)
    # proc.wait()

    # valueの学習
    # print("make valuemovedata")
    # procs = []
    # for gpu_idx in gpu_ids:
    #     proc = subprocess.Popen(["python3", "-m", "learn.reinforcement.make_valuemovedata", "--process_id", str(gpu_idx), "--gpu_ids", str(gpu_idx), "--value_batchsize", str(args.value_batchsize), "--value_batchnum", str(args.value_batchnum//len(gpu_ids)), "--output_path", value_movedata_path, "--learner_policy", new_learner_path, "--sl_policy", sl_path, "--temperature", str(args.temperature)], stdout=sys.stdout, stderr=sys.stdout)
    #     procs.append(proc)
    # [proc.wait() for proc in procs]

    # print("train valuenetwork")
    # # 試験的に、policy_movedataでvalueの学習もしてる
    # proc = subprocess.Popen(["python3", "-m", "learn.reinforcement.train_valuenetwork", "--movedata_path", policy_movedata_path, "--value_path", value_path, "--output_path", new_value_path, "--value_lr", str(args.value_lr)], stdout=sys.stdout, stderr=sys.stdout)
    # proc.wait()

    # 試験的に、policy_movedataでvalueの学習もしてる
    print("train networks")
    proc = subprocess.Popen(["python3", "-m", "learn.reinforcement.train_networks", "--movedata_path", policy_movedata_path, "--learner_path", learner_path, "--value_path", value_path, "--policy_output_path", new_learner_path, "--value_output_path", new_value_path, "--policy_lr", str(args.policy_lr), "--value_lr", str(args.value_lr)], stdout=sys.stdout, stderr=sys.stdout)
    proc.wait()

    # 勝率測定
    print("check_winrate")
    proc = subprocess.Popen(["python3", "-m", "learn.reinforcement.check_winrate", "--learner_path", new_learner_path, "--sl_path", sl_path, "--savedata_path", savedata_path, "--temperature", str(args.temperature)], stdout=sys.stdout, stderr=sys.stdout)
    proc.wait()

    if epoch % 10 == 9:
        # グラフ出力
        with open(savedata_path) as f:
            savedata = json.load(f)
            plt_winrates = savedata["wins"]
            plt_epochs = [epoch for epoch in range(savedata["len"])]

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ln1 = ax1.plot(plt_epochs, plt_winrates, "C0", label="win rate")
            h1, l1 = ax1.get_legend_handles_labels()
            ax1.legend(h1, l1)
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("win rate")
            fig.savefig(f"{args.output_path}/graphs/img_{epoch + 1}.png")