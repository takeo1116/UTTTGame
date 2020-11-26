# coding:utf-8

import os
import concurrent.futures
import random
from engine.game import Game

class RecordMaker():
    def make_game(self):
        # 先攻と後攻をランダムで1ゲーム作る
        players = random.sample(self.agent_names, 2)
        game = Game(players[0], players[1])
        return game

    def generate_record_file(self, battle_num, file_name):
        # battle_num回ゲームをシミュレートして、棋譜を記録する
        games = [self.make_game() for _ in range(battle_num)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
            futures = []
            for game in games:
                futures.append(executor.submit(game.play_for_record))
        
        # 各ゲームの棋譜をtxtファイルで出力する
        path = f"./{self.dir_name}/{file_name}.txt"
        with open(path, mode="w") as f:
            for future in futures:
                agent_names, record = future.result()
                for board, legal, player, move in record:
                    # boardは常に自分が1で相手が2になるように出力する
                    f.write(f"agent_name:{agent_names[player - 1]}\n")
                    f.write(f"board:{board if player == 1 else [[0, 2, 1][mark] for mark in board]}\n")
                    f.write(f"legal:{legal}\n")
                    f.write(f"move:{move}\n")
    
    def generate_records(self):
        os.mkdir(f"./{self.dir_name}")
        for idx in range(self.batch_num):
            self.generate_record_file(self.batch_size, f"record_{idx}")

    def __init__(self, player0, player1, batch_size, batch_num, dir_name):
        self.agent_names = [player0, player1]
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.dir_name = dir_name