# coding:utf-8

import os
import argparse
import torch
from torch._C import Value
from .recordmaker.valuedatamaker import ValuedataMaker
from learn.network.network import make_policynetwork

parser = argparse.ArgumentParser(description="valueデータの作成")
parser.add_argument("--batch_size", type=int,
                    default=100, help="1ファイルに記録するparallelgameの数")
parser.add_argument("--batch_num", type=int, default=1, help="ファイルをいくつ作成するか")
parser.add_argument("--save_path", type=str, default="", help="棋譜を保存するディレクトリのパス")
parser.add_argument("--parallel_num", type=int, default=128, help="並列で実行するゲームの数")
parser.add_argument("--policy_a", type=str, help="model_aのポリシーのpath")
parser.add_argument("--policy_b", type=str, help="model_bのポリシーのpath")

args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)

device = "cuda"
model_a = make_policynetwork()
model_a = model_a.to(device)
model_a = torch.nn.DataParallel(model_a)
model_a.load_state_dict(torch.load(args.policy_a), strict=False)

model_b = make_policynetwork()
model_b = model_b.to(device)
model_b = torch.nn.DataParallel(model_b)
model_b.load_state_dict(torch.load(args.policy_b), strict=False)

valuedata_maker = ValuedataMaker(model_a, model_b, args.batch_size, args.batch_num, args.save_path, args.parallel_num)
valuedata_maker.generate_valuedata()