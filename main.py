# coding:utf-8

import concurrent.futures
import os
from engine.game import Game
from record_maker import RecordMaker
from learning.record_processor import RecordProcessor

# 一戦
game = Game("SupervisedLearningAgent", "MctsAgent_1000")
game.play()
game.print_board()

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
#     "MixedAgent", "MixedAgent", 100, 10, "records/MixedAgent_vs_MixedAgent/MixedAgent_vs_MixedAgent_0")
# record_maker.generate_records()

# 棋譜読み込みテスト
# record_processor = RecordProcessor("./records", ["MctsAgent_1000"])
# print(record_processor.sample(5))
