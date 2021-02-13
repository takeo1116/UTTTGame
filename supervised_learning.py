# coding:utf-8

import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from learning.record_processor import RecordProcessor
from learning.learning_util import make_network

record_processor = RecordProcessor("./records/MixedAgent_vs_MixedAgent", ["MctsAgent_10000"])
dataLoader = DataLoader(record_processor.dataset, batch_size=1024, shuffle=True)
print(f"data_loaded:{len(dataLoader.dataset)}datas")

model = make_network()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

itr = 0
plt_iteration, plt_loss = [], []
def train(epoch):
    model.train()

    global itr
    global plt_iteration
    global plt_loss

    for board_tensor, move_tensor in dataLoader:
        optimizer.zero_grad()
        outputs = model(board_tensor)
        loss = loss_fn(outputs, move_tensor)
        # print(outputs)
        # print(move_tensor)
        plt_iteration.append(itr)
        plt_loss.append(loss.item())
        itr += 1
        loss.backward()
        optimizer.step()

    print(epoch)

for idx in range(100):
    train(idx)

model_path = "models/test2.pth"
# torch.save(model.state_dict(), model_path)

plt.plot(plt_iteration, plt_loss)
plt.savefig("img.png")