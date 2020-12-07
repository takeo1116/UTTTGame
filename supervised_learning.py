# coding:utf-8

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from record_processor import RecordProcessor

record_processor = RecordProcessor("./records/MCTS_1000_vs_Random", ["MctsAgent_1000"])
dataLoader = DataLoader(record_processor.dataset, batch_size=1, shuffle=True)
print(f"data_loaded:{len(dataLoader.dataset)}datas")

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

def eval():
    correct = 0
    total = 0
    model.eval()

    for board_tensor, move_tensor in dataLoader:
        outputs = model(board_tensor)
        _, predicted = torch.max(outputs.data, 1)
        if predicted == move_tensor:
            correct += 1
        total += 1
        print(predicted, move_tensor)
        print(correct, total)

# for idx in range(100):
#     train(idx)

# model_path = "models/test2.pth"
# torch.save(model.state_dict(), model_path)

# model.load_state_dict(torch.load(model_path))
# eval()