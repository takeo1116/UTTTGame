# coding:utf-8

class AgentBase:
    def request_move(self, board, legal, player_num):
        # 入力->盤面、合法手、自分のプレイヤー番号、出力->打つ場所
        return legal[0]
    def game_end(self, board, player_num, result):
        # ゲーム終了時に呼び出される関数
        # 入力->盤面、自分のプレイヤー番号、結果、結果
        pass
    def __init__(self):
        # エージェント生成時の処理
        pass