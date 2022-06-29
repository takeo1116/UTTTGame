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
import pickle
from setproctitle import setproctitle
from engine.parallelgame import ParallelGames
from engine.record import MoveDataEncoder
from learn.recordmaker.valuedatamaker import ValuedataMaker
from learn.util.feature import convert_movedata, make_feature, make_tensordatasetlist
from learn.util.pickmove import pick_legalmoves
from learn.network.network import make_pvnetwork
from learn.recordmaker.recordmaker import RecordMaker
from agent.neural_agent import NeuralAgent
from agent.neural_mcts import NeuralMctsAgent
from engine.game import Game

parser = argparse.ArgumentParser(description="モデル同士で対戦して、データを記録する")
parser.add_argument("--proc_name", type=str,
                    default="UTTT", help="プロセスの名前")
parser.add_argument("--output_path", type=str,
                    default="./RL_output", help="学習に使うディレクトリ")
parser.add_argument("--battle_num", type=int,
                    default=4, help="1度にバトルする数")
parser.add_argument("--bin_num", type=int,
                    default=10, help="学習データを分けるディレクトリの数")
parser.add_argument("--temperature", type=float,
                    default=0.01, help="modelが手を選択するときの温度")
args = parser.parse_args()

setproctitle(f"{args.proc_name}_battle")

print(f"temp: {args.temperature}")

# modelからモデルを読み込んで対戦させ、dataに入れていく
device = "cpu"
model_0 = make_pvnetwork(device=device)
model_1 = make_pvnetwork(device=device)
model_0.to(device)
model_1.to(device)

model_path = os.path.join(args.output_path, f"models")
while True:
    model_num = len(os.listdir(model_path))
    model0_idx = model_num - 1   # 最新
    # model1_idx = random.randrange(model_num)   # ランダム
    model1_idx = model_num - 1   # 最新

    movedatalist = []

    while True:
        try:
            model_0.load_state_dict(torch.load(os.path.join(model_path, f"state_{model0_idx}.pth"), map_location=torch.device(device)))
            model_1.load_state_dict(torch.load(os.path.join(model_path, f"state_{model1_idx}.pth"), map_location=torch.device(device)))
            break
        except:
            print("model load failed")
            time.sleep(10)
    
    # agent_0 = NeuralAgent(model_0, args.temperature)
    # agent_1 = NeuralAgent(model_1, args.temperature)
    agent_0 = NeuralMctsAgent(model_0, 500, args.temperature)
    agent_1 = NeuralMctsAgent(model_1, 500, args.temperature)

    print(f"model0: {model0_idx}, model1: {model1_idx}")

    win, lose = 0, 0

    # 0が先手
    for battle_idx in range(args.battle_num // 2):
        game = Game(agent_0, agent_1)
        movedatalist += game.play_for_record().record
        if game.game_state == 1:
            win += 1
        elif game.game_state == 2:
            lose += 1

    # 1が先手
    for battle_idx in range(args.battle_num // 2):
        game = Game(agent_1, agent_0)
        movedatalist += game.play_for_record().record
        if game.game_state == 2:
            win += 1
        elif game.game_state == 1:
            lose += 1

    print(len(movedatalist))
    datasetlist = make_tensordatasetlist(movedatalist, args.bin_num)

    print(f"winrate : model0win={win}, model1win={lose}, draw={args.battle_num - (win + lose)}")

    for idx, dataset in enumerate(datasetlist):
        with open(os.path.join(args.output_path, f"data/data_{idx}/dataset_{random.randrange(1e9)}"), "wb") as p:
            pickle.dump(dataset, p)