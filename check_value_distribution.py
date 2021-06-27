# coding:utf-8

import os
import torch
import json
import random
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from learning.record_processor import RecordProcessor
from learning.learning_util import make_valuenetwork
from learning.dataset_loader import DatasetLoader

class ValueDatasetLoader:
    def read_dataset_file(self, file_path):
        # file_pathで指定されたデータファイルを読んで、中身のデータを取り出す
        print(file_path)
        with open(file_path) as f:
            jsonStr = json.load(f)
        data = [(board, 1 if result == 0 else -1) for board, result in jsonStr]
        return data

    def read_dataset_folder(self, dir_path):
        # dir_path以下に入っているデータを再帰的に全部読む
        data = [] # [(board, result)]
        path_list = os.listdir(dir_path)
        dir_list = [os.path.join(dir_path, path) for path in path_list if os.path.isdir(
            os.path.join(dir_path, path))]
        file_list = [os.path.join(dir_path, path) for path in path_list if os.path.isfile(
            os.path.join(dir_path, path))]
        
        for file_path in file_list:
            data += self.read_dataset_file(file_path)
        
        for dir_path in dir_list:
            data += self.read_dataset_folder(dir_path)
        return data

    def read_datasets(self, dir_path):
        # dir_path以下のデータをすべて読んでdatasetを作る
        data = self.read_dataset_folder(dir_path)
        board_data, result_data = [], []

        for board, result in data:
            board_data.append(board)
            result_data.append([result])
        
        board_tensor = torch.Tensor(board_data).cuda()
        result_tensor = torch.Tensor(result_data).cuda()

        dataset = TensorDataset(board_tensor, result_tensor)
        return dataset

    def __init__(self, dir_path):
        self.dataset = self.read_datasets(dir_path) 

test_dir = "./valuedatasets/test/"
test_datasetLoader = ValueDatasetLoader(test_dir)
test_dataLoader = DataLoader(test_datasetLoader.dataset, batch_size=8192, shuffle=False)
print(f"loaded {len(test_datasetLoader.dataset)} test datas")

device = torch.device('cuda')
model = make_valuenetwork()
model = model.to(device)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("./models/value_learn_20.pth"))

# 読み込んだテストデータについて、正解ラベルが1のときと-1のときで予測値の分布を出力する
model.eval()
x = [(i - 50) / 50 for i in range(101)]
dist = [[] for _ in range(2)]
with torch.no_grad():
    for board_tensor, result_tensor in test_dataLoader:
        outputs = model(board_tensor)
        flatten = sum(outputs.tolist(), [])
        results = sum(result_tensor.tolist(), [])
        for out, res in zip(flatten, results):
            dist[0 if res > 0.0 else 1].append(out)
print(len(dist[0]), len(dist[1]))
fig = plt.figure()
plt.hist([dist[0], dist[1]], x, label=["win", "lose"])
plt.legend(loc="upper left")
fig.savefig("test.png")