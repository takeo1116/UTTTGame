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

parser = argparse.ArgumentParser(description="learnerとslを対戦させて勝率を得る")
parser.add_argument("--learner_path", type=str,
                    default=None, help="学習するポリシーのパス")
parser.add_argument("--sl_path", type=str,
                    default=None, help="slポリシーのパス")
parser.add_argument("--savedata_path", type=str,
                    default=None, help="savedataのパス")
parser.add_argument("--temperature", type=float,
                    default=1.0, help="modelが手を選択するときの温度")
args = parser.parse_args()

cpu_num = min(16, multiprocessing.cpu_count())

learner = make_policynetwork()
learner.cuda()
learner = torch.nn.DataParallel(learner)
print(f"load {args.learner_path}")
learner.load_state_dict(torch.load(args.learner_path), strict=False)

sl = make_policynetwork()
sl.cuda()
sl = torch.nn.DataParallel(sl)
print(f"load {args.sl_path}")
sl.load_state_dict(torch.load(args.sl_path), strict=False)

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

# セーブデータ更新
plt_winrates = []
with open(args.savedata_path) as f:
    savedata = json.load(f)
    plt_winrates = savedata["wins"]
    plt_winrates.append(learner_win / 10000)

with open(args.savedata_path, mode="w") as f:
    json.dump({"wins": plt_winrates, "len": len(plt_winrates)}, f)