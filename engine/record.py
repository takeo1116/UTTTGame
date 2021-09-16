# coding:utf-8
from enum import Enum


class MoveDataResult(Enum):
    # その手によって最終的に試合結果がどうなったか
    NOSET = 0
    WIN = 1
    LOSE = 2
    DRAW = 3
    ERROR = 4


class RecordResult(Enum):
    # 試合結果
    NOSET = 0
    PLAYER1WIN = 1
    PLAYER2WIN = 2
    DRAW = 3
    ERROR = 4


class MoveData:
    # 手
    # flat_boardは先手後手に関わらず空きが0, 自分が1, 相手が2
    def __init__(self, player_idx, is_first, agent_name, flat_board, legal_moves, move, result=MoveDataResult.NOSET):
        self.player_idx = player_idx
        self.is_first = is_first
        self.agent_name = agent_name
        self.flat_board = flat_board
        self.legal_moves = legal_moves
        self.move = move
        self.result = result


class Record:
    # ゲームの棋譜
    def append(self, move_data):
        self.record.append(move_data)

    # レコードの各MoveDataに結果を記録する
    def add_result(self, result=RecordResult.NOSET):
        for movedata in self.record:
            if result == RecordResult.PLAYER1WIN:
                movedata.result = MoveDataResult.WIN if movedata.player_idx == 1 else MoveDataResult.LOSE
            elif result == RecordResult.PLAYER2WIN:
                movedata.result = MoveDataResult.LOSE if movedata.player_idx == 1 else MoveDataResult.WIN
            elif result == RecordResult.DRAW:
                movedata.result = MoveDataResult.DRAW
            else:
                movedata.result = MoveDataResult.ERROR

    def __init__(self):
        self.record = []