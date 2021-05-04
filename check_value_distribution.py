# coding:utf-8

import torch
import json
import random
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from learning.record_processor import RecordProcessor
from learning.learning_util import make_network, make_value_network
from learning.dataset_loader import DatasetLoader

def make_dataset(file_paths):
    # jsonファイルを読み込んで、Datasetを作る
    board_data, result_data = [], []
    for file_path in file_paths:
        with open(file_path) as f:
            data = json.load(f)

        for board, result in data:
            board_data.append(board)
            result_data.append([1 if result == 0 else -1])

    print(len(result_data))

    board_tensor = torch.Tensor(board_data).cuda()
    result_tensor = torch.Tensor(result_data).cuda()

    dataset = TensorDataset(board_tensor, result_tensor)
    return dataset

test_path = "./valuedatasets/test/dataset_0.json"
batch_size = 1024

test_dataset = make_dataset([test_path])
test_dataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = make_value_network()
model.load_state_dict(torch.load("./models/value_transfer_fc_3000.pth"))

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
# print(dist)
print(len(dist[0]), len(dist[1]))
fig = plt.figure()
plt.hist([dist[0], dist[1]], x, label=["win", "lose"])
plt.legend(loc="upper left")
fig.savefig("test.png")