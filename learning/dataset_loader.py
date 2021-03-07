# coding:utf-8

import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader

class DatasetLoader():

    def load_dataset(self, input_path, input_filename):
        # jsonファイルを読み込んで、datasetを作る
        with open(f"{input_path}/{input_filename}") as f:
            data = json.load(f)

        board_data, move_data = [], []

        for board, move in data:
            board_data.append(board)
            move_data.append(move)

        board_tensor = torch.Tensor(board_data).cuda()
        move_tensor = torch.LongTensor(move_data).cuda()

        dataset = TensorDataset(board_tensor, move_tensor)
        return dataset

    def __init__(self, input_path, input_filename):
        self.dataset = self.load_dataset(input_path, input_filename)
