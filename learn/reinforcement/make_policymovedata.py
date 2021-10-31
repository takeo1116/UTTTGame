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

parser = argparse.ArgumentParser(description="ポリシーの対戦")
parser.add_argument("--process_id", type=int, default=0,
                    help="processID")
parser.add_argument("--gpu_ids", type=str, default=0, help="使用するgpuのid(カンマ区切り)")
parser.add_argument("--temperature", type=float,
                    default=1.0, help="modelが手を選択するときの温度")
parser.add_argument("--policy_batchsize", type=int,
                    default=256, help="一度に並列対戦する回数（policy）")
parser.add_argument("--policy_batchnum", type=int, default=250,
                    help="1epochあたりに何batch対戦するか（policy）")
parser.add_argument("--output_path", type=str,
                    default=None, help="実行結果をoutputするディレクトリ")
parser.add_argument("--learner_policy", type=str,
                    default=None, help="learnerのポリシー")
parser.add_argument("--enemy_policy_path", type=str,
                    default=None, help="enemyのポリシーが入っているディレクトリのパス")
args = parser.parse_args()

gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
cpu_num = min(16, multiprocessing.cpu_count())

# lernerとenemyを規定回数対戦させて、movedataを取りだす
device = torch.device(f"cuda:{gpu_ids[0]}")
learner = make_policynetwork()
learner.to(device)
learner = torch.nn.DataParallel(learner, device_ids=gpu_ids)
learner.load_state_dict(torch.load(args.learner_policy), strict=False)
learner.eval()
enemy = make_policynetwork()
enemy.to(device)
enemy = torch.nn.DataParallel(enemy, device_ids=gpu_ids)
enemy.eval()

def play_games(first_model, second_model, parallel_num):
    # modelどうしを対戦させて、movedatalistを取りだす
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_num) as executor:
        games = ParallelGames(parallel_num)
        player_idx, processing_boards = games.get_nowboards()
        while len(processing_boards) > 0:
            model, agent_name = (first_model, "first") if player_idx == 1 else (
                second_model, "second")
            flatboardlist = [
                flat_board for flat_board, _ in processing_boards]
            legalmoveslist = [legal_moves for _,
                                legal_moves in processing_boards]
            results = executor.map(make_feature, flatboardlist, legalmoveslist, chunksize=max(
                1024, len(processing_boards)//cpu_num))
            features = [feature for feature in results]
            feature_tensor = torch.Tensor(features).to(device)
            outputs = model(feature_tensor)
            moves = pick_legalmoves(
                outputs, [legal_moves for _, legal_moves in processing_boards], args.temperature)
            games.process_games(moves, agent_name)
            player_idx, processing_boards = games.get_nowboards()
        games.add_recordresults()
    return games

movedatalist = []
parallel_num = args.policy_batchsize // 2
enemy_num = len(os.listdir(args.enemy_policy_path))

for batch in range(args.policy_batchnum):
    enemy_idx = random.randrange(enemy_num)
    enemy_path = os.path.join(args.enemy_policy_path, f"policy_{enemy_idx}.pth")
    enemy.load_state_dict(torch.load(enemy_path))
    enemy.eval()
    games_A = play_games(learner, enemy, parallel_num)
    games_B = play_games(enemy, learner, parallel_num)

    win, lose = 0, 0
    win += sum([1 if state == 1 else 0 for state in games_A.game_states]) + sum([1 if state == 2 else 0 for state in games_B.game_states])
    lose += sum([1 if state == 2 else 0 for state in games_A.game_states]) + sum([1 if state == 1 else 0 for state in games_B.game_states])

    movedatalist += games_A.get_movedatalist() + games_B.get_movedatalist()
    print(f"vs enemy {enemy_idx} : win={win}, lose={lose}, draw={args.policy_batchsize - win - lose}")

with open(os.path.join(args.output_path, f"movedata_{args.process_id}.pth"), mode="w") as f:
    json.dump(movedatalist, f, cls=MoveDataEncoder)