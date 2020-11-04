# coding:utf-8

import abc
class AgentBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def request_move(self, board, legal, player_num):
        # 入力->盤面、合法手、自分のプレイヤー番号、出力->打つ場所
        pass
    @abc.abstractmethod
    def game_end(self, board, player_num, result):
        # ゲーム終了時に呼び出される関数
        # 入力->盤面、自分のプレイヤー番号、結果、結果
        pass
    @abc.abstractmethod
    def __init__(self):
        # エージェント生成時の処理
        pass