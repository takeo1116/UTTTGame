# coding:utf-8

from engine.record import MoveDataDecoder, MoveDataResult
import json
import concurrent.futures
from engine.game import Game
from agent.random_agent import RandomAgent
from agent.mcts_agent import MctsAgent
from agent.mixed_agent import MixedAgent

# players = [RandomAgent(), RandomAgent()]
# players = [MctsAgent(5000), MctsAgent(1000)]
players = [MixedAgent(), MixedAgent()]

# 一戦
# game = Game(players[0], players[1])
# result = game.play()
# print(result)
# for movedata in game.record.record:
#     print(f"player_idx = {movedata.player_idx}, is_first = {movedata.is_first}, agent_name = {movedata.agent_name}, flat_board = {movedata.flat_board}, legal_moves = {movedata.legal_moves}, move = {movedata.move}, result = {movedata.result}")

# 並列対戦
# games = [Game(players[0], players[1]) for _ in range(100)]
# with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
#     futures = []
#     for idx, game in enumerate(games):
#         futures.append(executor.submit(games[idx].play))

# record読み込み
path = "./test/record_0.json"
with open(path) as f:
    data = json.load(f, cls=MoveDataDecoder)
print(f"{data[0]}")
for movedata in data:
    print(movedata.result == MoveDataResult.WIN)
    print(f"player_idx = {movedata.player_idx}, is_first = {movedata.is_first}, agent_name = {movedata.agent_name}, flat_board = {movedata.flat_board}, legal_moves = {movedata.legal_moves}, move = {movedata.move}, result = {movedata.result}")
