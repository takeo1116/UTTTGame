# coding:utf-8

import os
import json
import torch
import random
import multiprocessing
import concurrent.futures
from torch import nn
from engine.parallelgame import ParallelGames
from engine.record import MoveDataEncoder
from learn.util.feature import make_feature


class ValuedataMaker:
    def play(self, r=random.randrange(40)):
        print(f"r={r}")

        def pick_legalmoves(outputs, legal_moves, temp=1.0):
            # 合法手のなかからsoftmaxで手を選択する
            # temp > 0.0
            beta = 1.0/temp
            softmax = nn.Softmax(dim=1).cuda()
            raw_probs = softmax(beta * outputs)
            probs = torch.Tensor([[prob + 0.001 if idx in legal else 0.0 for idx, prob in enumerate(
                probs)] for probs, legal in zip(raw_probs.tolist(), legal_moves)])
            moves = torch.multinomial(probs, 1).cuda()
            moves = moves.flatten()
            return moves.tolist()

        # 並列で1セットプレイする
        # r-1手目までをmodel_aで指して、r手目をランダムに、r+1手目以降を終局までmodel_bで指す
        # r+1手目のmovedataをlistにして返す（ランダムに打った直後の盤面の価値）
        games = ParallelGames(self.parallel_num)
        processing_boards = games.get_processing_boards()
        turn = 0
        cpu_num = min(16, multiprocessing.cpu_count())
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_num) as executor:
            while len(processing_boards) > 0:
                if turn == r:
                    agent_name = "Random"
                    moves = [random.choice(legal_moves)
                             for _, legal_moves in processing_boards]
                    games.process_games(moves, agent_name)
                else:
                    model, agent_name = (self.model_a, "Model_A") if turn < r else (
                        self.model_b, "Model_B")
                    model.eval()
                    flatboardlist = [
                        flat_board for flat_board, _ in processing_boards]
                    legalmoveslist = [legal_moves for _,
                                      legal_moves in processing_boards]
                    results = executor.map(
                        make_feature, flatboardlist, legalmoveslist, chunksize=max(1024, len(processing_boards)//cpu_num))
                    features = [feature for feature in results]
                    feature_tensor = torch.Tensor(features).cuda()
                    outputs = model(feature_tensor)
                    moves = pick_legalmoves(
                        outputs, [legal_moves for _, legal_moves in processing_boards], self.temp)
                    games.process_games(moves, agent_name)

                processing_boards = games.get_processing_boards()
                turn += 1
            games.add_recordresults()

        return [record.record[r+1] for record in games.records if len(record.record) > r+1]

    def generate_valuedata(self, rand_max=40):
        # parallel_num並列の対戦 × batch_size回を1ファイルにして、それをbatch_numファイル作る
        for idx in range(self.batch_num):
            movedatalist = sum([self.play(random.randrange(rand_max))
                                for _ in range(self.batch_size)], [])
            path = os.path.join(self.save_path, f"valuedata_{idx}.json")
            with open(path, mode="w") as f:
                json.dump(movedatalist, f, cls=MoveDataEncoder)

    def __init__(self, model_a, model_b, batch_size, batch_num, save_path, parallel_num=1, temp=1.0):
        self.model_a = model_a
        self.model_b = model_b
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.save_path = save_path
        self.parallel_num = parallel_num
        self.temp = temp
