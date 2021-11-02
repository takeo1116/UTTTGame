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

parser = argparse.ArgumentParser(description="バリューを1epoch学習する")
parser.add_argument("--movedata_path", type=str,
                    default=None, help="movedataのpath")
parser.add_argument("--value_path", type=str,
                    default=None, help="学習するバリューのパス")
parser.add_argument("--output_path", type=str,
                    default=None, help="valueを記録するパス")
parser.add_argument("--value_lr", type=float,
                    default=0.000001, help="valueの基本学習率")
args = parser.parse_args()

cpu_num = min(16, multiprocessing.cpu_count())

value = make_valuenetwork()
value.cuda()
value = torch.nn.DataParallel(value)
print(f"load {args.value_path}")
value.load_state_dict(torch.load(args.value_path), strict=False)

value_recordreader = RecordReader(args.movedata_path)
movedatalist = value_recordreader.get_movedatalist()

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

torch.save(value.state_dict(), args.output_path)