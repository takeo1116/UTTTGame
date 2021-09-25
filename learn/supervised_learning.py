# coding:utf-8

import torch
import os
import json
import random
import argparse
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


parser = argparse.ArgumentParser(description="棋譜データから教師あり学習を行う")
parser.add_argument("--records_path", type=str, default="./learn/records", help="学習する棋譜が入ったディレクトリ")
parser.add_argument("--output_path", type=str, default="./learn/SL_output", help="グラフ、モデルをoutputするディレクトリ")
parser.add_argument("--batch_size", type=int, default=8192, help="1バッチに含まれる教師データの数")
args = parser.parse_args()

# outputするディレクトリの作成
os.makedirs(args.output_path, exist_ok=True)
os.makedirs(os.path.join(args.output_path, "models"), exist_ok=True)
os.makedirs(os.path.join(args.output_path, "graphs"), exist_ok=True)
# os.makedirs(os.path.join(args.output_path, "log"), exist_ok=True)

# 棋譜の読み込み


# モデルの作成

# 学習とアウトプット