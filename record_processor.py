# coding:utf-8

import os

class RecordProcessor():
    def str_to_record(self, strings):
        agent_str, board_str, legal_str, move_str = strings[0], strings[1], strings[2], strings[3]
        agent = agent_str.lstrip("agent_name:")
        


    def record_to_data(self, record):
        # recordを加工して、学習に使える教師データに変換する
        _, board, legal, move = record
        legal_board = [1 if pos in legal else 0 for pos in range(81)]
        datum = board + legal_board
        return (datum, move)

    def read_record_file(self, file_path):
        # file_pathで指定されたファイルを読んで、中身のデータを取り出す
        with open(file_path) as f:
            row_records = f.read().split()
        # 4行ずつ読んで、(agent, board, legal, move)の形に整形する
        
        data = [self.record_to_data(record) for record in records]


    def read_records(self, dir_path):
        # dir_path以下に入っている棋譜ファイルを再帰的に全部読む
        data = []

        path_list = os.listdir(dir_path)
        dir_list = [path for path in path_list if os.path.isdir(os.path.join(dir_path, path))]
        file_list = [path for path in path_list if os.path.isfile(os.path.join(dir_path, path))]
        
        for file_path in file_list:
            self.read_record_file(file_path)

        for dir_path in dir_list:
            self.read_records(dir_path)

        return data

    def __init__(self, dir_path):
        self.data = []   # 局面と着手のリスト