# coding:utf-8

from learn.util.feature import make_dataloader
from learn.util.recordreader import RecordReader
from learn.util.rotation import distinct, multiply_movedatalist
import torch
import os
import argparse
import matplotlib.pyplot as plt
from torch import nn, optim
from learn.network.network import make_valuenetwork


def train():
    # 1 epoch学習させて、lossの平均値と訓練データに対するaccuracyを返す
    model.train()
    loss_sum = 0.0
    for feature_tensor, move_tensor, value_tensor in train_dataloader:
        optimizer.zero_grad()
        outputs = model(feature_tensor).cuda()
        loss = loss_fn(outputs, value_tensor).cuda()
        loss = loss.mean()
        loss_sum += loss.item() * feature_tensor.shape[0]
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)

    data_num = len(train_dataloader.dataset)
    loss_mean = loss_sum / data_num
    return loss_mean


def test():
    # テストデータに対するlossを返す
    model.eval()
    loss_sum = 0.0
    with torch.no_grad():
        for feature_tensor, move_tensor, value_tensor in test_dataloader:
            outputs = model(feature_tensor).cuda()
            loss = loss_fn(outputs, value_tensor).cuda()
            loss = loss.mean()
            loss_sum += loss.item() * feature_tensor.shape[0]

    data_num = len(test_dataloader.dataset)
    loss_mean = loss_sum / data_num
    return loss_mean


def test_distribution():
    # 勝ったデータと負けたデータでevaluationの分布を調べる
    model.eval()
    dist = [[] for _ in range(2)]
    with torch.no_grad():
        for feature_tensor, move_tensor, value_tensor in test_dataloader:
            outputs = model(feature_tensor).cuda()
            flatten = sum(outputs.tolist(), [])
            results = sum(value_tensor.tolist(), [])
            for out, res in zip(flatten, results):
                if res > 0.9:
                    dist[0].append(out)
                elif res < -0.9:
                    dist[1].append(out)
    return dist


parser = argparse.ArgumentParser(description="棋譜データからvalueの教師あり学習を行う")
parser.add_argument("--records_path", type=str,
                    default="./learn/records", help="学習する棋譜が入ったディレクトリ")
parser.add_argument("--output_path", type=str,
                    default="./learn/SL_output", help="グラフ、モデルをoutputするディレクトリ")
parser.add_argument("--batch_size", type=int,
                    default=8192, help="1バッチに含まれる教師データの数")
parser.add_argument("--epoch", type=int, default=10000, help="学習するepoch数")
parser.add_argument("--init_value", type=str, default=None, help="初期バリューのパス")
args = parser.parse_args()

# outputするディレクトリの作成
os.makedirs(args.output_path, exist_ok=True)
os.makedirs(os.path.join(args.output_path, "models"), exist_ok=True)
os.makedirs(os.path.join(args.output_path, "graphs"), exist_ok=True)
# os.makedirs(os.path.join(args.output_path, "log"), exist_ok=True)

# モデルの作成
device = "cuda"
model = make_valuenetwork()
model = model.to(device)
model = torch.nn.DataParallel(model)
if args.init_value is not None:
    model.load_state_dict(torch.load(args.init_value), strict=False)
    print(f"value {args.init_value} loaded")
loss_fn = nn.MSELoss(reduction="none")
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# 棋譜の読み込み
record_reader = RecordReader(args.records_path)
movedatalist = distinct((record_reader.get_movedatalist()))
train_dataloader, test_dataloader = make_dataloader(
    movedatalist, args.batch_size)

# 学習とアウトプット
plt_idx, plt_train_loss, plt_test_loss = [], [], []
for epoch in range(args.epoch):
    train_loss = train()
    test_loss = test()

    plt_idx.append(epoch)
    plt_train_loss.append(train_loss)
    plt_test_loss.append(test_loss)
    print(f"train_loss:{train_loss}, test_loss:{test_loss}")

    if epoch % 10 == 9:
        model_path = f"{args.output_path}/models/policy_{epoch + 1}.pth"
        torch.save(model.module.state_dict(), model_path)

        # lossのグラフ
        fig = plt.figure()
        plt.plot(plt_idx, plt_train_loss, plt_idx, plt_test_loss)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        fig.savefig(f"{args.output_path}/graphs/loss_{epoch + 1}.png")

        # 分布のグラフ
        dist = test_distribution()
        print(f"windata = {len(dist[0])}, losedata = {len(dist[1])}")
        fig = plt.figure()
        x = [(i - 50) / 50 for i in range(101)]
        print(dist)
        plt.hist([dist[0], dist[1]], x, label=["win", "lose"])
        plt.legend(loc="upper left")
        fig.savefig(f"{args.output_path}/graphs/distribution_{epoch + 1}.png")
