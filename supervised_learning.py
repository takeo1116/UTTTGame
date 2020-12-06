# coding:utf-8

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from record_processor import RecordProcessor

record_processor = RecordProcessor("./records/", ["MctsAgent_1000"])
dataLoader = DataLoader(record_processor.dataset, batch_size=1024, shuffle=True)

model = nn.Sequential()
model.add_module("fc1", nn.Linear(81*3, 100))
model.add_module("relu1", nn.ReLU())
model.add_module("fc2", nn.Linear(100, 100))
model.add_module("relu2", nn.ReLU())
model.add_module("fc3", nn.Linear(100, 81))

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
def train(epoch):
    model.train()

    for board_tensor, move_tensor in dataLoader:
        optimizer.zero_grad()
        outputs = model(board_tensor)
        loss = loss_fn(outputs, move_tensor)
        loss.backward()
        optimizer.step()
        # print(outputs)
        # print(move_tensor)

    print(epoch)


for idx in range(100):
    train(idx)