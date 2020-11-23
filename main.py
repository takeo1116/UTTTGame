# coding:utf-8

import threading
import concurrent.futures
from engine.game import Game

# 並列対戦
games = [Game("MctsAgent_1000", "RandomAgent") for _ in range(20)]
with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
    futures = []
    for idx, game in enumerate(games):
        print(f"start {idx}")
        futures.append(executor.submit(games[idx].play))