# coding:utf-8

import os
import concurrent.futures
import random
from engine.game import Game

class RecoredMaker():
    def play(self, idx):
        # 先攻と後攻をランダムで1ゲームシミュレートする
        players = random.sample(self.agent_names, 2)
        game = Game(players[0], players[1])
        game.play()
        self.records[idx] = (players, game.get_record())

    def make_record(self):
        # game_num回ゲームをシミュレートして、棋譜を記録する
        for idx in range(self.game_num):
            self.play(idx)

        os.mkdir(f"./{self.dir_name}")
            

    def __init__(self, player0, player1, game_num, dir_name):
        self.agent_names = [player0, player1]
        self.game_num = game_num
        self.dir_name = dir_name
        self.records = [None * game_num]