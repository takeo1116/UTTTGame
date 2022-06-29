# coding:utf-8

import random
import copy
import math
import torch
from .agentbase import AgentBase
from engine.board import Board
from learn.util.feature import make_feature
from learn.util.pickmove import pick_legalmoves
from torch import nn

class NeuralMctsAgent(AgentBase):
    def select_move(self, board_num, legal_moves, game_tree, value, policy, visited_num):
        C_BASE = 2
        # 手を選ぶ
        move, max_value = -1, -100000000.0
        total_num = visited_num[board_num]
        for pos in legal_moves:
            next_board_num = game_tree[board_num][pos]
            v, n = value[next_board_num]   # ない場合は0
            val = 0.0 if n <= 0 else v / n
            p = policy[board_num][pos]
            c = math.log((1 + total_num + C_BASE) / C_BASE) + 1
            val += c * p * math.sqrt(total_num) / (1 + visited_num[next_board_num])
            if max_value < val:
                move, max_value = pos, val
        return move

    def get_pv(self, flat_board, legal_moves):
        feature = make_feature(flat_board, legal_moves)
        feature_tensor = torch.Tensor([feature]).to(self.device)
        p, v = self.model(feature_tensor)
        return p, v

    def request_move(self, flat_board, legal_moves, is_first):
        # 現在の盤面から一定回数プレイアウトして、一番良い手を探す
        root_board = Board(flat_board)
        # game_tree[board_num][pos] = board_numの盤面で手番プレイヤーがposに打ったときに遷移するboardの番号
        game_tree = [[-1] * 81 for _ in range(self.TREE_SIZE)]
        # value[board_num] = (その盤面のスコアの合計, シミュレーションでそのboardにたどり着いた回数)
        value = [(0.0, 0)] * self.TREE_SIZE
        # policy[board_num][pos] = board_numの盤面でmodelがposに打つ確率
        policy = [[0.0] * 81 for _ in range(self.TREE_SIZE)]
        # visited_num[board_num] = そのboardを訪れた回数
        visited_num = [0] * self.TREE_SIZE
        board_count = 1
        for play_idx in range(self.playout_num):
            now_board = copy.deepcopy(root_board)
            now_player_num = 1  # request_moveに入ってくるflat_boardでは常に自分が1
            record = [(-1, -1, -1)]
            now_board_num, now_state = 0, 0

            while now_state == 0 and visited_num[now_board_num] > 0:
                now_legal = legal_moves if now_board_num == 0 else now_board.legal_moves(
                    record[-1][1])
                move = -1

                move = self.select_move(now_board_num, now_legal, game_tree, value, policy, visited_num)

                now_board.mark(move, now_player_num)
                record.append((now_board_num, move, now_player_num))
                now_board_num = game_tree[now_board_num][move]
                now_player_num = now_player_num ^ 3
                now_state = now_board.check_state()

            # 展開・フィードバック
            # value[0]は参照されないので更新する必要がない（してもよい）
            if now_state != 0:
                # 決着したとき
                visited_num[0] += 1
                for board_num, pos, player in record:
                    if board_num < 0:
                        continue
                    next_board_num = game_tree[board_num][pos]
                    visited_num[next_board_num] += 1
                    val, num = value[next_board_num]
                    value[next_board_num] = (val + 1.0, num + 1) if now_state == player else (val - 1.0, num + 1)
            elif visited_num[now_board_num] == 0:
                # 展開するとき
                now_legal = legal_moves if now_board_num == 0 else now_board.legal_moves(
                    record[-1][1])
                leaf_p, leaf_v = self.get_pv(now_board.flatten(), now_legal)
                value[now_board_num] = (-leaf_v[0][0], 1)   # 入れるのは「相手にその盤面を渡す価値」なので、盤面から推論した価値の逆
                visited_num[now_board_num] = 1
                prb = self.softmax(leaf_p)
                for pos in range(81):
                    game_tree[now_board_num][pos] = board_count
                    policy[now_board_num][pos] = prb[0][pos].item()
                    board_count += 1
                for board_num, pos, player in record:
                    if board_num < 0:
                        continue
                    visited_num[board_num] += 1
                    val, num = value[board_num]
                    value[board_num] = (val - leaf_v[0][0], num + 1) if now_player_num == player else (val + leaf_v[0][0], num + 1)
            else:
                raise Exception(f"exception in NeuralMctsAgent{(now_state, visited_num[now_board_num])}")

        # 最後に確率的に手を返す
        scores = torch.Tensor([[visited_num[game_tree[0][pos]] / self.playout_num for pos in range(81)]])
        beta = 1.0 / self.temperature
        raw_probs = self.softmax(beta * scores)
        probs = torch.Tensor([prob + 0.000001 if pos in legal_moves else 0.0 for pos, prob in enumerate(raw_probs[0].tolist())])
        moves = torch.multinomial(probs, 1)

        return moves.flatten()[0]

    def get_agentname(self):
        return f"NeuralMctsAgent_{self.playout_num}"

    def __init__(self, model, playout_num, temperature=0.01, device="cpu"):
        self.TREE_SIZE = 100000
        self.model = model
        self.playout_num = playout_num
        self.temperature = temperature
        self.device = device
        self.model.eval()
        self.softmax = nn.Softmax(dim=1).to(device)