# coding:utf-8

from learn.util.feature import make_dataloader
from learn.util.recordreader import RecordReader
from learn.util.rotation import distinct, multiply_movedatalist
import torch
import os
import argparse
import matplotlib.pyplot as plt
from torch import nn, optim
from learn.network.network import make_policynetwork


def train():
    # 1 epoch学習させて、lossの平均値と訓練データに対するaccuracyを返す
    model.train()
    correct = 0
    loss_sum = 0.0

    for feature_tensor, move_tensor, value_tensor in train_dataloader:
        optimizer.zero_grad()
        outputs = model(feature_tensor).cuda()
        loss = loss_fn(outputs, move_tensor).cuda()
        loss = loss.mean()
        loss_sum += loss.item() * feature_tensor.shape[0]
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(move_tensor.data.view_as(predicted)).sum()

    data_num = len(train_dataloader.dataset)
    loss_mean = loss_sum / data_num
    accuracy = correct / data_num
    return loss_mean, accuracy


def test():
    def count_legal(feature_tensor, predicted):
        # 合法手を返した数を調べる
        board_idxes = [0, 1, 2, 9, 10, 11, 18, 19, 20, 3, 4, 5, 12, 13, 14, 21, 22, 23, 6, 7, 8, 15, 16, 17, 24, 25, 26, 27, 28, 29, 36, 37, 38, 45, 46, 47, 30, 31, 32, 39,
                       40, 41, 48, 49, 50, 33, 34, 35, 42, 43, 44, 51, 52, 53, 54, 55, 56, 63, 64, 65, 72, 73, 74, 57, 58, 59, 66, 67, 68, 75, 76, 77, 60, 61, 62, 69, 70, 71, 78, 79, 80]
        move_list = predicted.tolist()
        legal = 0
        for idx, move in enumerate(move_list):
            board_idx = board_idxes[move]
            if feature_tensor[idx][2][board_idx // 9][board_idx % 9] > 0.5:
                legal += 1
        return legal

    # テストデータに対する正答率を返す
    model.eval()
    correct = 0
    legal = 0

    with torch.no_grad():
        for feature_tensor, move_tensor, value_tensor in test_dataloader:
            outputs = model(feature_tensor)

            _, predicted = torch.max(outputs.data, 1)  # 確率が最大のラベル
            # 当たってるものをカウント
            correct += predicted.eq(move_tensor.data.view_as(predicted)).sum()
            # 合法手をカウント
            legal += count_legal(feature_tensor, predicted)

    data_num = len(test_dataloader.dataset)
    accuracy = correct / data_num
    legal_rate = legal / data_num
    return accuracy, legal_rate


parser = argparse.ArgumentParser(description="棋譜データから教師あり学習を行う")
parser.add_argument("--records_path", type=str,
                    default="./learn/records", help="学習する棋譜が入ったディレクトリ")
parser.add_argument("--output_path", type=str,
                    default="./learn/SL_output", help="グラフ、モデルをoutputするディレクトリ")
parser.add_argument("--batch_size", type=int,
                    default=8192, help="1バッチに含まれる教師データの数")
parser.add_argument("--epoch", type=int, default=10000, help="学習するepoch数")
parser.add_argument("--init_policy", type=str, default=None, help="初期ポリシーのパス")
parser.add_argument("--teacher", type=str,
                    default="MctsAgent_10000", help="教師データにするエージェントの名前")
args = parser.parse_args()

# outputするディレクトリの作成
os.makedirs(args.output_path, exist_ok=True)
os.makedirs(os.path.join(args.output_path, "models"), exist_ok=True)
os.makedirs(os.path.join(args.output_path, "graphs"), exist_ok=True)
# os.makedirs(os.path.join(args.output_path, "log"), exist_ok=True)

# モデルの作成
device = "cuda"
model = make_policynetwork()
model = model.to(device)
model = torch.nn.DataParallel(model)
if args.init_policy is not None:
    model.load_state_dict(torch.load(args.init_policy), strict=False)
    print(f"policy {args.init_policy} loaded")
loss_fn = nn.CrossEntropyLoss(reduction="none")
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# 棋譜の読み込み
record_reader = RecordReader(args.records_path, args.teacher)
movedatalist = distinct(multiply_movedatalist(record_reader.get_movedatalist()))
train_dataloader, test_dataloader = make_dataloader(movedatalist, args.batch_size)

# 学習とアウトプット
plt_idx, plt_loss, plt_accuracy, plt_legal = [], [], [], []
for epoch in range(args.epoch):
    print(f"epoch {epoch} start")
    train_loss, train_accuracy = train()
    test_accuracy, legal_rate = test()
    plt_idx.append(epoch)
    plt_accuracy.append(test_accuracy)
    plt_loss.append(train_loss)
    plt_legal.append(legal_rate)
    print(f"loss:{train_loss}, train_accuracy:{train_accuracy}, test_accuracy:{test_accuracy}, legal:{legal_rate}")

    if epoch % 10 == 9:
        model_path = f"{args.output_path}/models/policy_{epoch + 1}.pth"
        torch.save(model.state_dict(), model_path)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ln1 = ax1.plot(plt_idx, plt_loss, "C0", label="train loss")
        ax2 = ax1.twinx()
        ln2 = ax2.plot(plt_idx, plt_accuracy, "C1", label="accuracy")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("train loss")
        ax1.set_ylim(0.5, 5.0)
        ax2.set_ylabel("accuracy")

        fig.savefig(f"{args.output_path}/graphs/img_{epoch + 1}.png")
