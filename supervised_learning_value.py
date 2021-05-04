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

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


train_dir = "./valuedatasets/train/"
test_path = "./valuedatasets/test/dataset_0.json"

batch_size = 1024

train_dataset = None
train_dataLoader = None
test_dataset = make_dataset([test_path])
test_dataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# model = make_network()
model = make_value_network()
# model.load_state_dict(torch.load("./models/alpha_50.pth"))
model.load_state_dict(torch.load("./models/value_learn_fc_3000.pth"))

# for param in model.parameters():
#     param.requires_grad = False
# model.fc1 = nn.Linear(in_features=model.channels_num * 9 * 9, out_features=256).cuda()
# model.fc2 = nn.Linear(in_features=256, out_features=1).cuda()

# torch.save(model.state_dict(), "models/value_init.pth")
# exit()

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)


def train(epoch, max_idx, sample_num):
    print(epoch)
    if epoch % 10 == 0:
        # ランダムに10データセットくらい選んでロードする
        global train_dataset
        global train_dataLoader
        idxes = random.sample([i for i in range(max_idx)], sample_num)
        print(idxes)
        file_paths = [f"{train_dir}dataset_{idx}.json" for idx in idxes]
        train_dataset = make_dataset(file_paths)
        train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    loss_sum = 0.0

    for board_tensor, result_tensor in train_dataLoader:
        optimizer.zero_grad()
        outputs = model(board_tensor).cuda()
        loss = loss_fn(outputs, result_tensor).cuda()
        loss_sum += loss.item() * board_tensor.shape[0]
        loss.backward()
        optimizer.step()

    data_num = len(train_dataset)
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

    data_num = len(test_dataset)
    loss_mean = loss_sum / data_num

    return loss_mean


plt_idx, plt_train_loss, plt_test_loss = [], [], []

for idx in range(10000):
    train_loss = train(idx, 84, 10)
    test_loss = test()

    plt_idx.append(idx)
    plt_train_loss.append(train_loss)
    plt_test_loss.append(test_loss)
    print(f"train_loss:{train_loss}, test_loss:{test_loss}")

    if idx % 100 == 99:
        model_path = f"models/value_learn_{idx + 1}.pth"
        torch.save(model.state_dict(), model_path)

        fig = plt.figure()
        plt.plot(plt_idx, plt_train_loss, plt_idx, plt_test_loss)
        plt.xlabel("epoch")
        plt.ylabel("loss")

        fig.savefig(f"img_{idx + 1}.png")

        