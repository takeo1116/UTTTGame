# coding:utf-8

import threading
import concurrent.futures
from engine.game import Game

# 一戦
# game = Game("MctsAgent_1000", "RandomAgent")
# game.play()
# game.print_board()

# 並列対戦
games = [Game("MctsAgent_1000", "RandomAgent") for _ in range(100)]
with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
    futures = []
    for idx, game in enumerate(games):
        futures.append(executor.submit(games[idx].play))