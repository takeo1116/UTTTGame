# coding:utf-8

import os
import json
from engine.record import MoveDataDecoder


class RecordReader:
    def read_records(self, records_path):
        def read_recordfile(file_path):
            # file_pathで指定された棋譜ファイルを読んで中身を取り出す
            with open(file_path) as f:
                datalist = json.load(f, cls=MoveDataDecoder)
            return [data for data in datalist if self.agentname is None or data.agent_name == self.agentname]

        # records_path以下の棋譜を再帰的に全部読み込む
        movedatalist = []
        paths = os.listdir(records_path)
        dirs = [os.path.join(records_path, path) for path in paths if os.path.isdir(
            os.path.join(records_path, path))]
        files = [os.path.join(records_path, path) for path in paths if os.path.isfile(
            os.path.join(records_path, path))]

        for file_path in files:
            movedatalist += read_recordfile(file_path)
        for dir_path in dirs:
            movedatalist += self.read_records(dir_path)
        return movedatalist

    def get_movedatalist(self):
        if self.movedatalist is None:
            self.movedatalist = self.read_records(self.records_path)
        print(f"RecordReader : read {len(self.movedatalist)} data")
        return self.movedatalist

    def __init__(self, records_path, agentname=None):
        self.records_path = records_path
        self.agentname = agentname
        self.movedatalist = None
