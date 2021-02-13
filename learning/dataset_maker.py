# coding:utf-8

import os
import random
import json
from .record_processor import RecordProcessor


class DatasetMaker(RecordProcessor):

    def board_to_string(self, board):
        # boardをstringに変換する
        string = ""
        flat_board = sum([sum(l, []) for l in board], [])
        for val in flat_board:
            string += "1" if val == 1 else "0"
        return string

    def output_datasets(self, output_path):
        # 訓練データとテストデータを分けて出力する
        os.mkdir(output_path)
        with open(f"{output_path}/train.json", mode="w") as f:
            json.dump(self.train, f)
        with open(f"{output_path}/test.json", mode="w") as f:
            json.dump(self.test, f)

    def read_records(self, input_path):
        # input_path以下の棋譜ファイルを全部読んで、同一の盤面を除いたdatasetを作る
        data = self.read_record_folder(input_path)
        used_boards = set()
        dataset = []

        for board, move in data:
            board_str = self.board_to_string(board)
            if board_str not in used_boards:
                dataset.append([board, move])
                used_boards.add(board_str)

        # シャッフルして、訓練データとテストデータを分ける
        random.shuffle(dataset)
        data_num = len(dataset)
        train_num = data_num - data_num // 10
        train, test = dataset[:train_num], dataset[train_num:]
        print(f"traindata : {len(train)}, testdata : {len(test)}")

        return (train, test)

    def __init__(self, input_path, agent_list):
        self.agent_list = agent_list  # 拾うエージェントのリスト
        self.train, self.test = self.read_records(input_path)
