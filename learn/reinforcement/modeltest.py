# coding:utf-8

from learn.util.feature import make_dataloader
from learn.util.recordreader import RecordReader
from learn.util.rotation import distinct, multiply_movedatalist
import torch
import os
import argparse
import matplotlib.pyplot as plt
from torch import nn, optim
from learn.network.network import make_pvnetwork
from setproctitle import setproctitle

parser = argparse.ArgumentParser(description="モデルのテストをする")
parser.add_argument("--proc_name", type=str,
                    default="UTTT", help="プロセスの名前")
parser.add_argument("--records_path", type=str,
                    default="./RL_output", help="モデルが入っているディレクトリ")
parser.add_argument("--models_dir", type=str,
                    default="./RL_output", help="モデルが入っているディレクトリ")
parser.add_argument("--teacher", type=str,
                    default="MctsAgent_10000", help="教師データにするエージェントの名前")
parser.add_argument("--batch_size", type=int,
                    default=8192, help="1バッチに含まれる教師データの数")
# parser.add_argument("--device", type=str,
#                     default="cuda", help="学習に使うデバイス")
args = parser.parse_args()

setproctitle(f"{args.proc_name}_modeltest")

# 棋譜の読み込み
record_reader = RecordReader(args.records_path, args.teacher)
movedatalist = distinct(multiply_movedatalist(record_reader.get_movedatalist()))
test_dataloader, _ = make_dataloader(movedatalist, args.batch_size)

# モデルの作成
device = "cuda"
model = make_pvnetwork()
model = model.to(device)

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
    dist = [[] for _ in range(2)]

    with torch.no_grad():
        for feature_tensor, move_tensor, value_tensor in test_dataloader:
            p, v = model(feature_tensor)

            # pについてのテスト
            _, predicted = torch.max(p.data, 1)  # 確率が最大のラベル
            # 当たってるものをカウント
            correct += predicted.eq(move_tensor.data.view_as(predicted)).sum()
            # 合法手をカウント
            legal += count_legal(feature_tensor, predicted)

            # vについてのテスト
            flatten = sum(v.tolist(), [])
            results = sum(value_tensor.tolist(), [])
            for out, res in zip(flatten, results):
                if res > 0.9:
                    dist[0].append(out)
                elif res < -0.9:
                    dist[1].append(out)

    data_num = len(test_dataloader.dataset)
    accuracy = correct / data_num
    legal_rate = legal / data_num
    return accuracy, legal_rate, dist


pathlist = os.listdir(args.models_dir)

for idx in range(len(pathlist)):
    # model
    model.load_state_dict(torch.load(os.path.join(args.models_dir, f"state_{idx}.pth")), strict=False)
    test_accuracy, legal_rate, dist = test()
    print(f"test_accuracy:{test_accuracy}, legal:{legal_rate}")