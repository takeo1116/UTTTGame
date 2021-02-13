# coding:utf-8

import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader

class DatasetLoader():

    def load_dataset(self, input_path):
        # jsonファイルを読み込んで、datasetを作る
        with open(input_path) as f:
            data = json.load(f)
        

    def __init__(self, input_path):
        self.train, self.test = self.load_dataset()
