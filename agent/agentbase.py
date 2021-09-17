# coding:utf-8

import abc


class AgentBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def request_move(self, flat_board, legal_moves, is_first):
        # 入力 -> (flat_board, 合法手, 先手かどうか), 出力 -> (打つ場所)
        return legal_moves[0]

    @abc.abstractmethod
    def get_agentname(self):
        # 棋譜に記録されるagent_nameを返す
        return "AgentBase"

    @abc.abstractmethod
    def __init__(self):
        # エージェント生成時の処理
        pass