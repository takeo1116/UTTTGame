# UTTTGame - Re:Dive

Ultimate Tic-Tac-Toeを使って遊ぶリポジトリ  
強化学習である程度強くなっていってそうなので、リファクタリングのためのブランチ。

開発日記　→　https://takeo1116.sakura.ne.jp/UTTT/UTTT.html

## ゲームエンジン
学習しやすいように自由に改造したいので自前で実装する。モデルの学習はPytorchで行うので、接続しやすいようにゲームエンジンもPythonを使用する（C#などで実装するよりも遅いが、まあ……）

- ゲームエンジンとエージェントの関係をどうするか？
- 学習のために、盤面のレコードを出力する機能の実装
    - （エージェントの名前、盤面の情報、合法手、指し手、勝敗）
    - 盤面の反転をゲームエンジン側で行う
        - エージェント呼び出し時に先手後手の情報を入力
        - 盤面情報は「自分が1, 相手が2」といった形で行う
- 強化学習のために、複数の盤面を並行して進める機能が必要
- 将来的にはGUIに対応させる？

## 教師あり学習
学習の1st STEPとして教師あり学習をする。UTTTにはプロ棋士はいないので、MCTSエージェントの指し手を教師データとする。  
テスト実装ではResNetを使って指し手の一致率60％を達成した。

```
UTTTGame$ python3 -m learn.make_record --batch_size 5 --batch_num 1 --save_path ./learn/records/test --parallel_num 15
```

## 強化学習
2nd STEPとして強化学習をする。まだテスト中だが強くなってる気がしたので先んじてReDiveを始めておく


## Usage
### 棋譜生成
```
UTTTGame$ python3 -m learn.make_record --batch_size 200 --batch_num 30 --save_path ./learn/records/test --parallel_num 15
```
### 教師あり学習
```
UTTTGame$ python3 -m learn.supervised_learning_policy --records_path ./learn/records --output_path ./learn/SL_output_test --epoch 20
```

### valuedata生成
```
UTTTGame$ python3 -m learn.make_valuedata --batch_size 200 --policy_a ./policy_a.pth --policy_b ./policy_b.pth --batch_num 30 --save_path ./learn/records/test --parallel_num 128
```

### value教師あり学習
```
UTTTGame$ python3 -m learn.supervised_learning_value --records_path ./learn/valuedata --output_path ./learn/SL_value_test --epoch 20
```

UTTTGame$ python3 -m learn.supervised_learning_value --records_path ./learn/records/valuetest --output_path ./learn/SL_value_test --epoch 20


UTTTGame$ python3 -m learn.make_valuedata --batch_size 200 --policy_a ./learn/SL_output_test/models/test_100.pth --policy_b ./learn/SL_output_test/models/test_100.pth --batch_num 1 --save_path ./learn/records/valuetest --parallel_num 128