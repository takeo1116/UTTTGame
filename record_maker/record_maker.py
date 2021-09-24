# coding:utf-8

import os
import json
import copy
import concurrent.futures
import random
from engine.game import Game
from engine.record import MoveDataEncoder


class RecordMaker:
    def make_game(self):
        # 先後ランダムで1ゲーム作る
        players = random.sample(self.agents, 2)
        game = Game(copy.deepcopy(players[0]), copy.deepcopy(players[1]))
        return game

    def generate_recordfile(self, battle_num, file_name):
        # battle_num回ゲームをシミュレートして、棋譜を記録する
        games = [self.make_game() for _ in range(battle_num)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.parallel_num) as executor:
            futures = []
            for game in games:
                futures.append(executor.submit(game.play_for_record))

        # 各ゲームの棋譜をjsonに出力していく
        movedatalist = []
        for future in futures:
            try:
                record = future.result()
                for movedata in record.record:
                    movedatalist.append(movedata)
            except Exception as e:
                print("exception occured in future")
                print(type(e))
                print(e)

        path = os.path.join(self.save_path, f"{file_name}.json")
        with open(path, mode="w") as f:
            json.dump(movedatalist, f, cls=MoveDataEncoder)

    def generate_records(self):
        for idx in range(self.batch_num):
            self.generate_recordfile(self.batch_size, f"record_{idx}")

    def __init__(self, agent_1, agent_2, batch_size, batch_num, save_path, parallel_num=1):
        self.agents = [agent_1, agent_2]
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.save_path = save_path
        self.parallel_num = parallel_num
