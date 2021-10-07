# coding:utf-8
from engine.record import MoveData, Record, RecordResult
from .board import Board


class ParallelGames:
    # 複数のゲームをまとめて処理する
    def make_recordresult(self, game_state):
        return [RecordResult.NOSET, RecordResult.PLAYER1WIN, RecordResult.PLAYER2WIN, RecordResult.DRAW][game_state]

    def make_requestboard(self, flat_board, player_idx):
        request_board = [mark for mark in flat_board] if player_idx == 1 else [[0, 2, 1][mark] for mark in flat_board]
        return request_board

    def get_processing_boards(self):
        # 進行中の盤面のリストを取得する
        # 全ゲーム終了していたら空リストを返す
        return [(self.make_requestboard(self.boards[idx].flatten(), self.now_player), self.boards[idx].legal_moves(self.prev_moves[idx])) for idx, state in enumerate(self.game_states) if state == 0]

    def get_processing_boards_idx(self):
        # 進行中のBoardとindexのリストを取得する
        return [(idx, self.boards[idx], self.boards[idx].legal_moves(self.prev_moves[idx])) for idx, state in enumerate(self.game_states) if state == 0]

    def get_nowboards(self):
        # 次に打つプレイヤーと現在の盤面を取得する（終わってないもののみ）
        return (self.now_player, self.get_processing_boards())

    def process_games(self, moves, agent_name="default"):
        # 手の一覧を入力することで盤面を進める
        boards_idx = self.get_processing_boards_idx()
        is_first = (self.now_player == self.first_player)
        if len(boards_idx) != len(moves):
            raise Exception(
                f"ParallelGame.process_games : len(processing_boards) = {len(boards_idx)} and len(moves) = {len(moves)}")
        for (idx, board, legal_moves), move in zip(boards_idx, moves):
            if move in legal_moves:
                board.mark(move, self.now_player)
                self.game_states[idx] = board.check_state()
                flat_board = board.flatten()
                request_board = self.make_requestboard(flat_board, self.now_player)
                self.records[idx].append(MoveData(
                    self.now_player, is_first, agent_name, request_board, legal_moves, move))
                self.prev_moves[idx] = move
                self.game_states[idx] = board.check_state()
            else:
                self.game_states[idx] = self.now_player ^ 3
        self.now_player = self.now_player ^ 3
        return True

    def add_recordresults(self):
        for idx in range(self.parallel_num):
            self.records[idx].add_result(
                self.make_recordresult(self.game_states[idx]))

    def get_movedatalist(self, player_idx=-1):
        # movedataのリストを取得する（playerの指定可能）
        movedatalist = sum([record.record for record in self.records], [])
        if player_idx < 0:
            return movedatalist
        else:
            return [movedata for movedata in movedatalist if movedata.player_idx == player_idx]

    def __init__(self, parallel_num):
        self.parallel_num = parallel_num
        self.boards = [Board() for _ in range(parallel_num)]
        self.first_player = 1
        self.now_player = 1
        self.prev_moves = [-1 for _ in range(parallel_num)]
        self.game_states = [0 for _ in range(parallel_num)]
        self.records = [Record() for _ in range(parallel_num)]
