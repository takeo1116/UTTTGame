# coding:utf-8

import copy
import multiprocessing
import torch.multiprocessing as mp
import os
import shutil
import concurrent.futures
import random
import torch
import argparse
import json
from engine.parallelgame import ParallelGames
from engine.record import MoveDataEncoder
from learn.recordmaker.valuedatamaker import ValuedataMaker
from learn.util.feature import convert_movedata, make_feature
from learn.util.pickmove import pick_legalmoves
from learn.network.network import make_policynetwork, make_valuenetwork

parser = argparse.ArgumentParser(description="バリューの対戦")
parser.add_argument("--process_id", type=int, default=0,
                    help="processID")
parser.add_argument("--gpu_ids", type=str, default=0, help="使用するgpuのid(カンマ区切り)")
parser.add_argument("--temperature", type=float,
                    default=1.0, help="modelが手を選択するときの温度")
parser.add_argument("--value_batchsize", type=int,
                    default=8000, help="一度に並列対戦する回数（value）")
parser.add_argument("--value_batchnum", type=int, default=5,
                    help="1epochあたりに何batch対戦するか（value）")
parser.add_argument("--output_path", type=str,
                    default=None, help="実行結果をoutputするディレクトリ")
parser.add_argument("--learner_policy", type=str,
                    default=None, help="learnerのポリシー")
parser.add_argument("--sl_policy", type=str,
                    default=None, help="slのポリシー")
args = parser.parse_args()

gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]

device = torch.device(f"cuda:{gpu_ids[0]}")
learner = make_policynetwork()
learner.to(device)
learner = torch.nn.DataParallel(learner, device_ids=gpu_ids)
learner.load_state_dict(torch.load(args.learner_policy), strict=False)
learner.eval()
sl = make_policynetwork()
sl.to(device)
sl = torch.nn.DataParallel(sl, device_ids=gpu_ids)
sl.load_state_dict(torch.load(args.sl_policy), strict=False)
sl.eval()

movedatalist = []
valuedatamaker = ValuedataMaker(sl, learner, args.value_batchnum, 1, "no save", args.value_batchsize, args.temperature)
movedatalist = valuedatamaker.get_valuedata(70)

with open(os.path.join(args.output_path, f"movedata_{args.process_id}.pth"), mode="w") as f:
    json.dump(movedatalist, f, cls=MoveDataEncoder)