# coding:utf-8

import torch
from record_processor import RecordProcessor
# とりあえず、学習してモデルを更新する機能をつくる
record_processor = RecordProcessor("./records", ["MctsAgent_1000"])

board_data, label_data = record_processor.sample(5)
board_tensors, label_tensors = torch.Tensor(board_data), torch.Tensor(label_data)

print(board_tensors)
print(label_tensors)