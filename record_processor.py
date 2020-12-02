# coding:utf-8

import os
import json
import random

class RecordProcessor():

    def convert_record(self, record):
        # recordを加工して、学習に使える教師データに変換する
        board, legal, move = record["board"], record["legal"], record["move"]
        legal_board = [1 if pos in legal else 0 for pos in range(81)]
        datum = board + legal_board
        return (datum, move)

    def read_record_file(self, file_path):
        # file_pathで指定されたファイルを読んで、中身のデータを取り出す
        with open(file_path) as f:
            records = json.load(f)
        data = [self.convert_record(record) for record in records if record["agent"] in self.agent_list]
        return data

    def read_records(self, dir_path):
        # dir_path以下に入っている棋譜ファイルを再帰的に全部読む
        data = []
        path_list = os.listdir(dir_path)
        dir_list = [os.path.join(dir_path, path) for path in path_list if os.path.isdir(
            os.path.join(dir_path, path))]
        file_list = [os.path.join(dir_path, path) for path in path_list if os.path.isfile(
            os.path.join(dir_path, path))]

        for file_path in file_list:
            data += self.read_record_file(file_path)

        for dir_path in dir_list:
            data += self.read_records(dir_path)
        return data

    def sample(self, num):
        # num個のデータをランダムに取り出す
        return random.sample(self.data, num)

    def __init__(self, dir_path, agent_list):
        self.agent_list = agent_list    # 拾うエージェントのリスト
        self.data = self.read_records(dir_path)   # 局面と着手のリスト
