# coding:utf-8

import os
import json
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from learning.learning_util import convert_record


class RecordProcessor():

    def read_record_file(self, file_path):
        # file_pathで指定された棋譜ファイルを読んで、中身のデータを取り出す
        with open(file_path) as f:
            records = json.load(f)
        data = [convert_record(
            record) for record in records if record["agent"] in self.agent_list]
        return data

    def read_record_folder(self, dir_path):
        # dir_path以下に入っている棋譜ファイルを再帰的に全部読む
        data = []   # [(board, move)]
        path_list = os.listdir(dir_path)
        dir_list = [os.path.join(dir_path, path) for path in path_list if os.path.isdir(
            os.path.join(dir_path, path))]
        file_list = [os.path.join(dir_path, path) for path in path_list if os.path.isfile(
            os.path.join(dir_path, path))]

        for file_path in file_list:
            data += self.read_record_file(file_path)

        for dir_path in dir_list:
            data += self.read_record_folder(dir_path)
        return data

    def read_records(self, dir_path):
        # dir_path以下の棋譜ファイルを全部読んでdatasetを作る
        data = self.read_record_folder(dir_path)
        board_data, move_data = [], []

        for board, move in data:
            board_data.append(board)
            move_data.append(move)

        board_tensor = torch.Tensor(board_data)
        move_tensor = torch.LongTensor(move_data)

        dataset = TensorDataset(board_tensor, move_tensor)
        return dataset

    def __init__(self, dir_path, agent_list):
        self.agent_list = agent_list    # 拾うエージェントのリスト
        self.dataset = self.read_records(dir_path)   # 局面と着手のリスト
