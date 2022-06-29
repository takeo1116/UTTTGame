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

parser = argparse.ArgumentParser(description="モデル同士で対戦して、データを記録する")
parser.add_argument("--proc_name", type=str,
                    default="UTTT", help="プロセスの名前")
parser.add_argument("--output_path", type=str,
                    default="./RL_output", help="学習に使うディレクトリ")
parser.add_argument("--battle_num", type=int,
                    default=500, help="1度にバトルする数の半分")
parser.add_argument("--bin_num", type=int,
                    default=100, help="学習データを分けるディレクトリの数")
parser.add_argument("--temperature", type=float,
                    default=1.0, help="modelが手を選択するときの温度")
args = parser.parse_args()

setproctitle(f"{args.proc_name}_battle")

print(f"temp: {args.temperature}")

cpu_num = min(16, multiprocessing.cpu_count())

# modelからモデルを読み込んで対戦させ、dataに入れていく
device = "cpu"
model_0 = make_pvnetwork(device=device)
model_1 = make_pvnetwork(device=device)
model_0.to(device)
model_1.to(device)

def play_games(first_model, second_model, parallel_num):
    # modelどうしを対戦させて、movedatalistを取りだす
    first_model.eval()
    second_model.eval()

    turn = 0
    r = random.randrange(10)

    games = ParallelGames(parallel_num)
    player_idx, processing_boards = games.get_nowboards()
    while len(processing_boards) > 0:
        if turn == r:
            agent_name = "Random"
            moves = [random.choice(legal_moves)
                        for _, legal_moves in processing_boards]
            games.process_games(moves, agent_name)
        else:
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
        turn += 1
    games.add_recordresults()

    return games

model_path = os.path.join(args.output_path, f"models")

while True:
    model_num = len(os.listdir(model_path))
    model0_idx = model_num - 1   # 最新
    model1_idx = random.randrange(model_num)   # ランダム
    # model1_idx = model_num - 1   # 最新

    movedatalist = []

    while True:
        try:
            model_0.load_state_dict(torch.load(os.path.join(model_path, f"state_{model0_idx}.pth"), map_location=torch.device(device)))
            model_1.load_state_dict(torch.load(os.path.join(model_path, f"state_{model1_idx}.pth"), map_location=torch.device(device)))
            break
        except:
            print("model load failed")
            time.sleep(10)
    
    print(f"model0: {model0_idx}, model1: {model1_idx}")
    games_A = play_games(model_0, model_1, args.battle_num)
    games_B = play_games(model_1, model_0, args.battle_num)

    movedatalist += games_A.get_movedatalist(player_idx=1)
    movedatalist += games_B.get_movedatalist(player_idx=2)
    # movedatalist += games_A.get_movedatalist()
    # movedatalist += games_B.get_movedatalist()

    print(len(movedatalist))
    datasetlist = make_tensordatasetlist(movedatalist, args.bin_num)

    win = sum([1 if state == 1 else 0 for state in games_A.game_states]) + sum([1 if state == 2 else 0 for state in games_B.game_states])
    lose = sum([1 if state == 2 else 0 for state in games_A.game_states]) + sum([1 if state == 1 else 0 for state in games_B.game_states])

    print(f"winrate : model0win={win}, model1win={lose}, draw={2 * args.battle_num - (win + lose)}")

    for idx, dataset in enumerate(datasetlist):
        with open(os.path.join(args.output_path, f"data/data_{idx}/dataset_{random.randrange(1e9)}"), "wb") as p:
            pickle.dump(dataset, p)