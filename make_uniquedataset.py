# coding:utf-8

from learning.dataset_maker import DatasetMaker

# 指定した場所に入っている棋譜ファイルから重複する盤面を取り除き、訓練データとテストデータを分ける

dataset_maker = DatasetMaker("./records/MixedAgent_vs_MixedAgent", ["MctsAgent_10000"])
dataset_maker.output_datasets("./datasets/MctsAgent_10000")