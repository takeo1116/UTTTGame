# coding:utf-8

import os
import torch
import json
import random
import argparse
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from learning.learning_util import make_policynetwork, make_valuenetwork

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

torch.manual_seed(0)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--dataset_path", type=str, help="データセットのパス", default="./valuedatasets/")
parser.add_argument("--output_path", type=str, help="pthを出力するパス", default="./models/")
parser.add_argument("--batch_size", type=int, help="バッチサイズ", default=8192)
args = parser.parse_args()
batch_size = args.batch_size
dataset_path = args.dataset_path
output_path = args.output_path

test_datasetLoader = ValueDatasetLoader(dataset_path + "test/")
test_dataLoader = DataLoader(test_datasetLoader.dataset, batch_size=batch_size, shuffle=True)
print(f"loaded {len(test_datasetLoader.dataset)} test datas")

train_datasetLoader = ValueDatasetLoader(dataset_path + "train/")
train_dataLoader = DataLoader(train_datasetLoader.dataset, batch_size=batch_size, shuffle=True)
print(f"loaded {len(train_datasetLoader.dataset)} train datas")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = make_policynetwork()
model = make_valuenetwork()
model = model.to(device)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("valuedata_policies/RL_init.pth"), strict=False)
# for param in model.parameters():
#     param.requires_grad = False
# model.module.fc = nn.Linear(in_features=512, out_features=9*9).cuda()
# model.module.out = nn.Linear(in_features=81, out_features=1).cuda()
# torch.save(model.state_dict(), "models/value_init.pth")
# exit()

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)


def train(epoch):
    print(epoch)
    model.train()
    loss_sum = 0.0

    for board_tensor, result_tensor in train_dataLoader:
        optimizer.zero_grad()
        outputs = model(board_tensor).cuda()
        loss = loss_fn(outputs, result_tensor).cuda()
        loss_sum += loss.item() * board_tensor.shape[0]
        loss.backward()
        optimizer.step()

    data_num = len(train_datasetLoader.dataset)
    loss_mean = loss_sum / data_num

    return loss_mean

def test():
    # テストデータに対するlossを返す
    model.eval()
    loss_sum = 0.0
    with torch.no_grad():
        for board_tensor, result_tensor in test_dataLoader:
            outputs = model(board_tensor)

            loss = loss_fn(outputs, result_tensor).cuda()
            loss_sum += loss.item() * board_tensor.shape[0]

    data_num = len(test_datasetLoader.dataset)
    loss_mean = loss_sum / data_num

    return loss_mean

plt_idx, plt_train_loss, plt_test_loss = [], [], []

for idx in range(10000):
    train_loss = train(idx)
    test_loss = test()

    plt_idx.append(idx)
    plt_train_loss.append(train_loss)
    plt_test_loss.append(test_loss)
    print(f"train_loss:{train_loss}, test_loss:{test_loss}")

    if idx % 20 == 19:
        model_path = f"models/value_learn_{idx + 1}.pth"
        torch.save(model.state_dict(), model_path)

        fig = plt.figure()
        plt.plot(plt_idx, plt_train_loss, plt_idx, plt_test_loss)
        plt.xlabel("epoch")
        plt.ylabel("loss")

        fig.savefig(f"img_{idx + 1}.png")