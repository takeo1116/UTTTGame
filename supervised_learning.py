# coding:utf-8

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from learning.record_processor import RecordProcessor
from learning.learning_util import make_network

record_processor = RecordProcessor("./records/MCTS_1000_vs_Random", ["MctsAgent_1000"])
dataLoader = DataLoader(record_processor.dataset, batch_size=1024, shuffle=True)
print(f"data_loaded:{len(dataLoader.dataset)}datas")

model = make_network()
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

    print(epoch)

for idx in range(100):
    train(idx)

model_path = "models/test2.pth"
torch.save(model.state_dict(), model_path)