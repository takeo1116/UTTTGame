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

parser = argparse.ArgumentParser(description="ポリシーを1epoch学習する")
parser.add_argument("--movedata_path", type=str,
                    default=None, help="movedataのpath")
parser.add_argument("--learner_path", type=str,
                    default=None, help="学習するポリシーのパス")
parser.add_argument("--value_path", type=str,
                    default=None, help="バリューのパス")
parser.add_argument("--output_path", type=str,
                    default=None, help="ポリシーを記録するパス")
parser.add_argument("--policy_lr", type=float,
                    default=0.0000001, help="policyの基本学習率")
args = parser.parse_args()

cpu_num = min(16, multiprocessing.cpu_count())

learner = make_policynetwork()
learner.cuda()
learner = torch.nn.DataParallel(learner)
print(f"load {args.learner_path}")
learner.load_state_dict(torch.load(args.learner_path), strict=False)

value = make_valuenetwork()
value.cuda()
value = torch.nn.DataParallel(value)
print(f"load {args.value_path}")
value.load_state_dict(torch.load(args.value_path), strict=False)

policy_recordreader = RecordReader(args.movedata_path)
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

torch.save(learner.state_dict(), args.output_path)