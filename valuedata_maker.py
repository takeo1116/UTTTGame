# coding:utf-8

import os
import json
import concurrent.futures
import random
import torch
from collections import OrderedDict
from engine.game_parallel import GameParallel
from learning.learning_util import make_network, make_value_network, convert_board, convert_record, pick_moves, pick_legal_moves

class ValueDataMaker():
    def generate_data(self):
        # 0から69まで適当な数を決めて、その手数までmodel_Aで打ち、1手ランダムに打つ
        r = random.randrange(70)
        game_parallel = GameParallel(1024)
        now_player, board_infos = game_parallel.get_nowboards()
        turn = 0
        while len(board_infos) > 0 and turn < r:
            # flat_boardsを変形してTensorにする
            board_data = [convert_board(flat_board, legal_moves)
                        for flat_board, legal_moves in board_infos]
            boards_tensor = torch.Tensor(board_data).cuda()
            outputs = self.agent_A(boards_tensor)

            moves = pick_legal_moves(
                outputs, [legal for _, legal in board_infos]) if now_player == 1 else pick_legal_moves(outputs, [legal for _, legal in board_infos])
            game_parallel.process_games(moves)
            
            turn += 1
            now_player, board_infos = game_parallel.get_nowboards()
        
        # ゲームが終わってないものに対して、1手ランダムに打つ
        if len(board_infos) > 0:
            rand_moves = [random.choice(legal_moves) for _, legal_moves in board_infos]
            game_parallel.process_games(rand_moves)
            now_player, board_infos = game_parallel.get_nowboards()

        # 終局までmodel_Bで打つ
        while len(board_infos) > 0:
            board_data = [convert_board(flat_board, legal_moves)
                        for flat_board, legal_moves in board_infos]
            boards_tensor = torch.Tensor(board_data).cuda()
            outputs = self.agent_B(boards_tensor)

            moves = pick_legal_moves(
                outputs, [legal for _, legal in board_infos]) if now_player == 1 else pick_legal_moves(outputs, [legal for _, legal in board_infos])
            game_parallel.process_games(moves)

            now_player, board_infos = game_parallel.get_nowboards()

        # 棋譜からr手目の盤面を勝敗の情報を取り出して、教師データとする
        data = []
        result_and_record = game_parallel.get_results_and_records_turn(r + 1)
        for result, record in result_and_record:
            player_idx, _, board, legal, _ = record
            # print(f"r={r}")
            # print(f"result = {result}, player_idx = {player_idx}")
            board_data = convert_board(board if player_idx == 1 else [[0, 2, 1][mark] for mark in board], legal)
            resnum = 0 if player_idx == result else 1
            # print(player_idx, result, resnum)
            data.append((board_data, resnum))
        return data

    def generate_dataset_file(self, file_name):
        # batch_size回シミュレートして結果を出力する
        dataset = []
        for i in range(self.batch_size):
            data = self.generate_data()
            dataset.extend(data)
        
        path = f"./{self.dir_name}/{file_name}.json"
        with open(path, mode="w") as f:
            json.dump(dataset, f)

    def generate_datasets(self):
        os.mkdir(f"./{self.dir_name}")
        for idx in range(self.batch_num):
            print(f"batch_num = {idx}")
            self.generate_dataset_file(f"dataset_{idx}")

    def __init__(self, model_A, model_B, batch_size, batch_num, dir_name):
        self.agent_A = make_network()
        self.agent_A.load_state_dict(torch.load(model_A))
        self.agent_B = make_network()
        self.agent_B.load_state_dict(torch.load(model_B))
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.dir_name = dir_name
        
