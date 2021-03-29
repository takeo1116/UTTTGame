# coding:utf-8

import concurrent.futures
import random
from engine.game import Game

# ゲームをbattle_num回シミュレートして、勝率を計算する
BATTLE_NUM = 100
agent_names = ["SupervisedLearningAgent", "MctsAgent_5000"]
def make_game(agent_names):
    # 先攻と後攻を固定で1ゲーム作る
    game = Game(agent_names[0], agent_names[1])
    return game

games = [make_game(agent_names) for _ in range(BATTLE_NUM)]
reversed_games = [make_game(agent_names[::-1]) for _ in range(BATTLE_NUM)]

win_1_a, win_2_a, draw_a, error_a = 0, 0, 0, 0
for game in games:
    data = game.play()
    if data.count("(player 1)") > 0:
        win_1_a += 1
    elif data.count("(player 2)") > 0:
        win_2_a += 1
    elif data == "draw":
        draw_a += 1
    else:
        error_a += 1

win_1_b, win_2_b, draw_b, error_b = 0, 0, 0, 0
for game in reversed_games:
    data = game.play()
    if data.count("(player 1)") > 0:
        win_1_b += 1
    elif data.count("(player 2)") > 0:
        win_2_b += 1
    elif data == "draw":
        draw_b += 1
    else:
        error_b += 1

# ↓並列
# with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
#     futures = []
#     reversed_futures = []
#     for game in games:
#         futures.append(executor.submit(game.play))
#     for game in reversed_games:
#         reversed_futures.append(executor.submit(game.play))

# win_1_a, win_2_a, draw_a, error_a = 0, 0, 0, 0
# for future in futures:
#     try:
#         data = future.result()
#         if data.count("(player 1)") > 0:
#             win_1_a += 1
#         elif data.count("(player 2)") > 0:
#             win_2_a += 1
#         elif data == "draw":
#             draw_a += 1
#         else:
#             error_a += 1

#     except Exception as e:
#         error_a += 1

# win_1_b, win_2_b, draw_b, error_b = 0, 0, 0, 0
# for future in reversed_futures:
#     try:
#         data = future.result()
#         print(f"!{data}")
#         if data.count("(player 1)") > 0:
#             win_1_b += 1
#         elif data.count("(player 2)") > 0:
#             win_2_b += 1
#         elif data == "draw":
#             draw_b += 1
#         else:
#             error_b += 1

#     except Exception as e:
#         error_b += 1
# ↑並列

print("ordered")
print(f"{agent_names[0]} wins {win_1_a}, {agent_names[1]} wins {win_2_a}, {draw_a} draws, {error_a} errors")
print("reversed")
print(f"{agent_names[1]} wins {win_1_b}, {agent_names[0]} wins {win_2_b}, {draw_b} draws, {error_b} errors")

