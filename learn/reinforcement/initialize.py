# coding:utf-8

import torch
import os
import argparse
import json
from setproctitle import setproctitle

from learn.network.network import make_pvnetwork

device = "cuda"

parser = argparse.ArgumentParser(description="強化学習の初期化を行う")
parser.add_argument("--proc_name", type=str,
                    default="UTTT", help="プロセスの名前")
parser.add_argument("--output_path", type=str,
                    default="./RL_output", help="学習に使うディレクトリ")
parser.add_argument("--bin_num", type=int,
                    default=100, help="学習データを分けるディレクトリの数")
parser.add_argument("--init_model", type=str,
                    default=None, help="学習の初期状態")       
args = parser.parse_args()

setproctitle(f"{args.proc_name}_initialize")

# ディレクトリを作成する
os.makedirs(args.output_path, exist_ok=False)
os.makedirs(os.path.join(args.output_path, "models"), exist_ok=True)
os.makedirs(os.path.join(args.output_path, "logs"), exist_ok=True)

savedata = {"epoch": 0}

# モデルの初期状態を作成する
model_path = os.path.join(args.output_path, "models")
os.makedirs(model_path, exist_ok=True)
model = make_pvnetwork()
model = model.to(device)
if args.init_model is not None:
    model.load_state_dict(torch.load(args.init_model), strict=False)

torch.save(model.state_dict(), os.path.join(model_path, "state_0.pth"))

# 対戦データを入れるディレクトリを作成する
for idx in range(args.bin_num):
    bin_path = os.path.join(args.output_path, f"data/data_{idx}")
    os.makedirs(bin_path)

with open(os.path.join(args.output_path, "savedata.json"), mode="w") as f:
    json.dump(savedata, f)
