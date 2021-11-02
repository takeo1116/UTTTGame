# coding:utf-8

import copy
import multiprocessing
import os
import shutil
import concurrent.futures
import random
import torch
import argparse
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from engine.parallelgame import ParallelGames
from learn.recordmaker.valuedatamaker import ValuedataMaker
from learn.util.feature import convert_movedata, make_feature
from learn.util.pickmove import pick_legalmoves


class ReinfocementTrainer():
    def play_parallelgames(self, parallel_num, first_model, second_model, temp=1.0, first_name="first", second_name="second"):
        # parallel_num並列対戦をしてgamesを返す
        first_model.eval()
        second_model.eval()
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.cpu_num) as executor:
            games = ParallelGames(parallel_num)
            player_idx, processing_boards = games.get_nowboards()
            while len(processing_boards) > 0:
                model, agent_name = (first_model, first_name) if player_idx == 1 else (
                    second_model, second_name)
                flatboardlist = [
                    flat_board for flat_board, _ in processing_boards]
                legalmoveslist = [legal_moves for _,
                                  legal_moves in processing_boards]
                results = executor.map(make_feature, flatboardlist, legalmoveslist, chunksize=max(
                    1024, len(processing_boards)//self.cpu_num))
                features = [feature for feature in results]
                feature_tensor = torch.Tensor(features).cuda()
                outputs = model(feature_tensor)
                moves = pick_legalmoves(
                    outputs, [legal_moves for _, legal_moves in processing_boards], temp)
                games.process_games(moves, agent_name)

                player_idx, processing_boards = games.get_nowboards()
            games.add_recordresults()
        return games

    def get_policy_dataloader(self, batch_size, batch_num, temp=1.0):
        # policyを規定回数対戦させてdataloaderを得る
        self.learner.eval()
        self.enemy.eval()
        movedatalist = []
        for _ in range(batch_num):
            self.change_enemy()
            win, lose = 0, 0
            for idx, models in enumerate([(self.learner, self.enemy, "learner", "enemy"), (self.enemy, self.learner, "enemy", "learner")]):
                learner_idx = idx + 1
                first_model, second_model, first_name, second_name = models
                games = self.play_parallelgames(
                    batch_size // 2, first_model, second_model, temp, first_name, second_name)
                movedatalist += games.get_movedatalist()
                win += sum([1 if state ==
                            learner_idx else 0 for state in games.game_states])
                lose += sum([1 if state == learner_idx ^
                            3 else 0 for state in games.game_states])
            print(f"win = {win}, lose = {lose}, draw = {batch_size - win - lose}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.cpu_num) as executor:
            results = executor.map(
                convert_movedata, movedatalist, chunksize=len(movedatalist)//self.cpu_num)
            result_list = [result for result in results]
            feature_tensor = torch.Tensor(
                [feature for feature, _, _ in result_list]).cuda()
            move_tensor = torch.LongTensor([move for _, move, _ in result_list]).cuda()
            rewards_tensor = torch.Tensor(
                [reward for _, _, reward in result_list]).cuda()
            dataset = TensorDataset(
                feature_tensor, move_tensor, rewards_tensor)
            dataloader = DataLoader(dataset, 8192, shuffle=True)
        return dataloader

    def train_policy(self, dataloader, lr):
        # policyを1エポック学習する
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        optimizer = optim.Adagrad(self.learner.parameters(), lr=lr)
        self.value.eval()
        self.learner.train()
        for feature_data, move_data, reward_data in dataloader:
            # 推論
            optimizer.zero_grad()
            outputs = self.learner(feature_data).cuda()

            # valueを推論する
            value_outputs = self.value(feature_data).cuda()
            value_data = sum(value_outputs.tolist(), [])
            coef_tensor = torch.Tensor(
                [(reward - value) for reward, value in zip(sum(reward_data.tolist(), []), value_data)]).cuda()

            loss = loss_fn(outputs, move_data).cuda()
            loss *= coef_tensor
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        return

    def get_value_dataloader(self, batch_size, batch_num, temp=1.0):
        # learnerを規定回数対戦させてdataloaderを得る(value)
        self.value.eval()
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.cpu_num) as executor:
            valuedatamaker = ValuedataMaker(
                self.slpolicy, self.learner, batch_num, 1, "no save", batch_size, temp)
            movedatalist = valuedatamaker.get_valuedata(70)

            results = executor.map(
                convert_movedata, movedatalist, chunksize=len(movedatalist)//self.cpu_num)
            result_list = [result for result in results]
            feature_tensor = torch.Tensor(
                [feature for feature, _, _ in result_list]).cuda()
            value_tensor = torch.Tensor(
                [value for _, _, value in result_list]).cuda()
            dataset = TensorDataset(
                feature_tensor, value_tensor)
            dataloader = DataLoader(dataset, 8192, shuffle=True)
        return dataloader

    def train_value(self, dataloader, lr):
        # valueを1エポック学習する
        self.value.train()
        loss_fn = nn.MSELoss(reduction="none")
        optimizer = optim.SGD(self.value.parameters(), lr=lr)
        for feature_tensor, value_tensor in dataloader:
            optimizer.zero_grad()
            outputs = self.value(feature_tensor).cuda()
            loss = loss_fn(outputs, value_tensor).cuda()
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        return

    def check_winrate(self, batch_size, temp=1.0):
        # learnerとslpolicyを対戦させて勝率を調べる
        # (learner_win, sl_win, draw)
        self.learner.eval()
        self.slpolicy.eval()
        learner_win, sl_win = 0, 0
        for idx, models in enumerate([(self.learner, self.slpolicy, "learner", "slpolicy"), (self.slpolicy, self.learner, "slpolicy", "learner")]):
            learner_idx = idx + 1
            first_model, second_model, first_name, second_name = models
            games = self.play_parallelgames(
                batch_size // 2, first_model, second_model, temp, first_name, second_name)
            learner_win += sum([1 if state ==
                                learner_idx else 0 for state in games.game_states])
            sl_win += sum([1 if state == learner_idx ^
                           3 else 0 for state in games.game_states])
        return (learner_win, sl_win, batch_size - (learner_win + sl_win))

    def change_enemy(self):
        # enemyを別のポリシーに更新する
        enemy_idx = random.randrange(0, len(self.enemy_statedicts))
        state_dict = self.enemy_statedicts[enemy_idx]
        self.enemy.load_state_dict(state_dict)
        print(f"load enemy {enemy_idx}")

    def save_lerner(self):
        # learnerのポリシーをセーブして返す
        state_dict = copy.deepcopy(self.learner.state_dict())
        self.enemy_statedicts.append(state_dict)
        return state_dict

    def __init__(self, learner, enemy, slpolicy, value, cpu_num):
        self.learner = learner
        self.enemy = enemy
        self.slpolicy = slpolicy
        self.value = value
        self.enemy_statedicts = []
        self.cpu_num = cpu_num
