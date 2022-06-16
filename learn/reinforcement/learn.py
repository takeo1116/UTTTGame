# coding:utf-8

import copy
import multiprocessing
import torch.multiprocessing as mp
import os
import shutil
import concurrent.futures
import random
import torch
from torch import nn, optim
import argparse
import json
import pickle
import time
from setproctitle import setproctitle
from torch.utils.data import TensorDataset, DataLoader
from engine.parallelgame import ParallelGames
from engine.record import MoveDataEncoder
from learn.recordmaker.valuedatamaker import ValuedataMaker
from learn.util.feature import convert_movedata, make_feature, make_tensordatasetlist
from learn.util.pickmove import pick_legalmoves
from learn.network.network import make_pvnetwork

parser = argparse.ArgumentParser(description="対戦データの学習を行う")
parser.add_argument("--proc_name", type=str,
                    default="UTTT", help="プロセスの名前")
parser.add_argument("--device", type=str,
                    default="cuda", help="学習に使うデバイス")
parser.add_argument("--input_path", type=str,
                    default="./RL_output", help="学習に使うディレクトリ")
parser.add_argument("--bin_num", type=int,
                    default=100, help="学習データを分けるディレクトリの数")
parser.add_argument("--bin_threshold", type=int,
                    default=50000, help="学習するデータ量の閾値")
parser.add_argument("--bin_max", type=int,
                    default=150000, help="学習を打ち切るデータ量の閾値")
parser.add_argument("--batch_size", type=int,
                    default=2500, help="1バッチに含まれる教師データの数の目安")
parser.add_argument("--remove_rate", type=float,
                    default=1.1, help="学習データを消す確率")
parser.add_argument("--lr", type=float,
                    default=0.0000001, help="学習率")
args = parser.parse_args()

setproctitle(f"{args.proc_name}_learn")

model_path = os.path.join(args.input_path, f"models")

model = make_pvnetwork()
model = model.to(args.device)

# 一番新しいmodelを読み込む
model_path = os.path.join(args.input_path, f"models")
model_num = len(os.listdir(model_path))
model.load_state_dict(torch.load(os.path.join(model_path, f"state_{model_num - 1}.pth")))

loss_fn_p = nn.CrossEntropyLoss(reduction="none")
loss_fn_v = nn.MSELoss(reduction="none")
optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

def train(datasets):
    def merge_datasets(datasets):
        features, moves, rewards = [], [], []
        for dataset in datasets:
            for feature_tensor, move_tensor, reward_tensor in dataset:
                features.append(feature_tensor)
                moves.append(move_tensor)
                rewards.append(reward_tensor)
            
        features_tensor = torch.stack(features).to(args.device)
        moves_tensor = torch.stack(moves).to(args.device)
        rewards_tensor = torch.stack(rewards).to(args.device)

        # print(features_tensor.shape, moves_tensor.shape, rewards_tensor.shape)

        return TensorDataset(features_tensor, moves_tensor, rewards_tensor)

    dataset = merge_datasets(datasets)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True) # 学習時にサイズが1だとbnで落ちるのでdropする

    model.train()

    for feature_tensor, move_tensor, reward_tensor in dataloader:
        optimizer.zero_grad()
        p, v = model(feature_tensor)
        value_data = sum(v.tolist(), [])
        coef_tensor = torch.Tensor([(reward - val) for reward, val in zip(sum(reward_tensor.tolist(), []), value_data)]).to(args.device)

        loss_p = loss_fn_p(p, move_tensor).to(args.device)
        loss_p * coef_tensor
        loss_v = loss_fn_v(v, reward_tensor).to(args.device)
        loss = loss_p.mean() + loss_v.mean()
        loss.backward()
        optimizer.step()


for epoch in range(1000000):
    # 順番に読み込んで学習していく
    for bin_idx in range(args.bin_num):
        print(f"epoch: {epoch}, bin: {bin_idx}")
        bin_path = os.path.join(args.input_path, f"data/data_{bin_idx}")
        learned = False
        while not learned:
            datapaths = [os.path.join(bin_path, path) for path in os.listdir(bin_path)]
            datasets = []
            bin_size = 0
            for path in datapaths:
                with open(path, "rb") as p:
                    dataset = pickle.load(p)
                    datasets.append(dataset)
                    bin_size += len(dataset)
                    if args.bin_max < bin_size:
                        print(f"stop reading")
                        break
            
            if bin_size < args.bin_threshold:
                print(f"not learned ({bin_size} data)")
                time.sleep(30)
            else:
                train(datasets)
                for removepath in datapaths:
                    r = random.random()
                    if r < args.remove_rate:
                        os.remove(removepath)
                print(f"learned ({bin_size} data)")
                break

    torch.save(model.state_dict(), os.path.join(model_path, f"state_{model_num}.pth"))
    model_num += 1