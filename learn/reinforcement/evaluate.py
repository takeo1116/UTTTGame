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

parser = argparse.ArgumentParser(description="モデルの評価を行う")
parser.add_argument("--proc_name", type=str,
                    default="UTTT", help="プロセスの名前")
parser.add_argument("--input_path", type=str,
                    default="./RL_output", help="学習に使うディレクトリ")
parser.add_argument("--eval_model", type=str,
                    default=None, help="対戦相手")
parser.add_argument("--temperature", type=float,
                    default=1.0, help="modelが手を選択するときの温度")
args = parser.parse_args()

setproctitle(f"{args.proc_name}_evaluate")

def play_games(first_model, second_model, parallel_num):
    # modelどうしを対戦させて、movedatalistを取りだす
    first_model.eval()
    second_model.eval()

    games = ParallelGames(parallel_num)
    player_idx, processing_boards = games.get_nowboards()
    while len(processing_boards) > 0:
        model, agent_name = (first_model, "first") if player_idx == 1 else (
            second_model, "second")
        flatboardlist = [
            flat_board for flat_board, _ in processing_boards]
        legalmoveslist = [legal_moves for _,
                            legal_moves in processing_boards]
        results = [make_feature(flatboard, legalmoves) for flatboard, legalmoves in zip(flatboardlist, legalmoveslist)]
        features = [feature for feature in results]
        feature_tensor = torch.Tensor(features).to(device)
        p, v = model(feature_tensor)
        moves = pick_legalmoves(
            p, [legal_moves for _, legal_moves in processing_boards], args.temperature)
        games.process_games(moves, agent_name)
        player_idx, processing_boards = games.get_nowboards()
    games.add_recordresults()

    return games

device = "cpu"
model_0 = make_pvnetwork(device=device)
model_1 = make_pvnetwork(device=device)
model_0.to(device)
model_1.to(device)
model_1.load_state_dict(torch.load(args.eval_model), strict=False)
model_path = os.path.join(args.input_path, f"models")

while True:
    model_num = len(os.listdir(model_path))
    model0_idx = model_num - 1   # 最新

    print(f"load model: {model_num - 1}")

    movedatalist = []

    model_0.load_state_dict(torch.load(os.path.join(model_path, f"state_{model0_idx}.pth"), map_location=torch.device(device)))

    games_A = play_games(model_0, model_1, 2500)
    games_B = play_games(model_1, model_0, 2500)

    win = sum([1 if state == 1 else 0 for state in games_A.game_states]) + sum([1 if state == 2 else 0 for state in games_B.game_states])
    lose = sum([1 if state == 2 else 0 for state in games_A.game_states]) + sum([1 if state == 1 else 0 for state in games_B.game_states])

    print(f"check winrate : win={win}, lose={lose}, draw={5000 - (win + lose)}")