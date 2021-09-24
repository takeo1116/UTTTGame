# coding:utf-8

import os
import argparse
from agent.mixed_agent import MixedAgent
from .record_maker import RecordMaker


parser = argparse.ArgumentParser(description="棋譜データの作成")
parser.add_argument("--batch_size", type=int,
                    default=100, help="1ファイルに記録する試合数")
parser.add_argument("--batch_num", type=int, default=1, help="ファイルをいくつ作成するか")
parser.add_argument("--save_path", default="", help="棋譜を保存するディレクトリのパス")
parser.add_argument("--parallel_num", type=int, default="1", help="並列で実行するゲームの最大数")
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)

agents = [MixedAgent(), MixedAgent()]
record_maker = RecordMaker(
    agents[0], agents[1], args.batch_size, args.batch_num, args.save_path, args.parallel_num)
record_maker.generate_records()
