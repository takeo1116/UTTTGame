# coding:utf-8

import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from learning.record_processor import RecordProcessor
from learning.learning_util import make_network
from learning.dataset_loader import DatasetLoader

# 乱数のseedを固定
torch.manual_seed(11)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_path = "./datasets/MctsAgent_10000"
train_datasetLoader = DatasetLoader(input_path, "train.json")
test_datasetLoader = DatasetLoader(input_path, "test.json")

batch_size = 1024
train_dataLoader = DataLoader(
    train_datasetLoader.dataset, batch_size=batch_size, shuffle=True)
test_dataLoader = DataLoader(
    test_datasetLoader.dataset, batch_size=batch_size, shuffle=False)
print(
    f"data loaded : {len(train_dataLoader.dataset)} train datas and {len(test_datasetLoader.dataset)} test datas")

model = make_network()
# model.load_state_dict(torch.load("./models/test_10000.pth"))   # 初期値をロードするとき
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

board_idxes = [0, 1, 2, 9, 10, 11, 18, 19, 20, 3, 4, 5, 12, 13, 14, 21, 22, 23, 6, 7, 8, 15, 16, 17, 24, 25, 26, 27, 28, 29, 36, 37, 38, 45, 46, 47, 30, 31, 32, 39,
               40, 41, 48, 49, 50, 33, 34, 35, 42, 43, 44, 51, 52, 53, 54, 55, 56, 63, 64, 65, 72, 73, 74, 57, 58, 59, 66, 67, 68, 75, 76, 77, 60, 61, 62, 69, 70, 71, 78, 79, 80]

def legal_count(board_tensor, predicted):
    # 合法手を返した数を調べる
    # board_tensorの形が変わったら変えること
    move_list = predicted.tolist()
    legal = 0
    for idx, move in enumerate(move_list):
        board_idx = board_idxes[move]
        if board_tensor[idx][2][board_idx // 9][board_idx % 9] > 0.5:
            legal += 1
    return legal

def train(epoch):
    # lossの平均値を返す
    model.train()
    loss_sum = 0

    for board_tensor, move_tensor in train_dataLoader:
        optimizer.zero_grad()
        outputs = model(board_tensor).cuda()
        loss = loss_fn(outputs, move_tensor).cuda()
        loss_sum += loss.item() * board_tensor.shape[0]
        loss.backward()
        optimizer.step()

    print(epoch)
    data_num = len(train_dataLoader.dataset)
    loss_mean = loss_sum / data_num
    return loss_mean


def test():
    # テストデータに対する正答率を返す
    model.eval()
    correct = 0
    legal = 0

    with torch.no_grad():
        for board_tensor, move_tensor in test_dataLoader:
            outputs = model(board_tensor)

            _, predicted = torch.max(outputs.data, 1)  # 確率が最大のラベル
            # 当たってるものをカウント
            correct += predicted.eq(move_tensor.data.view_as(predicted)).sum()
            # 合法手をカウント
            legal += legal_count(board_tensor, predicted)

    data_num = len(test_dataLoader.dataset)
    accuracy = correct / data_num
    legal_rate = legal / data_num

    return accuracy, legal_rate


plt_idx, plt_loss, plt_accuracy, plt_legal = [], [], [], []

for idx in range(200):
    loss_sum = train(idx)
    accuracy, legal_rate = test()
    plt_idx.append(idx)
    plt_accuracy.append(accuracy)
    plt_loss.append(loss_sum)
    plt_legal.append(legal_rate)
    print(f"loss:{loss_sum}, accuracy:{accuracy}, legal:{legal_rate}")

    if idx % 50 == 49:
        model_path = f"models/test_{idx + 1}.pth"
        torch.save(model.state_dict(), model_path)

print(plt_legal)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1 = ax1.plot(plt_idx, plt_loss, "C0", label="train loss")
ax2 = ax1.twinx()
ln2 = ax2.plot(plt_idx, plt_accuracy, "C1", label="accuracy")
# ln3 = ax2.plot(plt_idx, plt_legal, "C2", label="legal_rate")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2)
ax1.set_xlabel("epoch")
ax1.set_ylabel("train loss")
ax1.set_ylim(1.5, 5.0)
ax2.set_ylabel("accuracy")

fig.savefig("img.png")
