# coding:utf-8

import concurrent.futures
import os
import torch
from engine.game import Game
from record_maker import RecordMaker
from valuedata_maker import ValueDataMaker
from learning.record_processor import RecordProcessor
from learning.learning_util import make_value_network, convert_board


def print_board(flat_board):
    # flat_boardが表す盤面をコンソールに出力する
    nums = [0, 1, 2, 9, 10, 11, 18, 19, 20]
    for row in range(9):
        string = ""
        for idx, num in enumerate(nums):
            string += str(flat_board[num + 3 * row + row // 3 * 18])
            if idx in [2, 5]:
                string += " "
        print(string)
        if row in [2, 5]:
            print()


# 一戦
# game = Game("MctsAgent_5000", "MctsAgent_5000")
game = Game("SupervisedLearningAgent", "MctsAgent_2000")
# game = Game("SupervisedLearningAgent", "SupervisedLearningAgent")
records = game.play_for_record()
# game.print_board()

# 棋譜から評価値を出す
value_net = make_value_network()
# value_net.load_state_dict(torch.load("./models/transfer_2200.pth"))
value_net.load_state_dict(torch.load("./models/value_learn_fc_3000.pth"))
value_net.eval()
for player_idx, agent_name, flat_board, legal, move in records:
    print(f"player_idx = {player_idx}, agent_name = {agent_name}")
    print(f"legal = {legal}")
    print()
    print_board(flat_board)
    print(f"move = {move}")
    print()

    board = flat_board if player_idx == 1 else [[0, 2, 1][mark] for mark in flat_board]
    converted = convert_board(board, legal)
    board_tensor = torch.Tensor([converted]).cuda()
    value_outputs = value_net(board_tensor).cuda()
    print(f"value = {value_outputs.item()}")

# record = game.get_record()
# for rec in record:

#     print(rec)

# 並列対戦
# games = [Game("MctsAgent_1000", "RandomAgent") for _ in range(100)]
# with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
#     futures = []
#     for idx, game in enumerate(games):
#         futures.append(executor.submit(games[idx].play))

# 棋譜生成
# record_maker = RecordMaker(
#     "MixedAgent", "MixedAgent", 100, 100, "records/MixedAgent_vs_MixedAgent/MixedAgent_vs_MixedAgent_9")
# record_maker.generate_records()

# 棋譜読み込みテスト
# record_processor = RecordProcessor("./records", ["MctsAgent_1000"])
# print(record_processor.sample(5))

# value生成
# valuedata_maker = ValueDataMaker("./models/alpha_50.pth", "./models/alpha_50.pth", 100, 100, "valuedatasets/valuedata_0")
# valuedata_maker.generate_datasets()