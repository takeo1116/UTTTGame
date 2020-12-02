# coding:utf-8

import concurrent.futures
from engine.game import Game
from record_maker import RecordMaker

# 一戦
# game = Game("MctsAgent_1000", "RandomAgent")
# game.play()
# game.print_board()

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
record_maker = RecordMaker("MctsAgent_1000", "RandomAgent", 100, 10, "MCTS_1000_vs_Random_test_2")
record_maker.generate_records()