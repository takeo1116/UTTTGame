# UTTTGame - Re:Dive

Ultimate Tic-Tac-Toeを使って遊ぶリポジトリ  
強化学習である程度強くなっていってそうなので、リファクタリングのためのブランチ。

開発日記　→　https://takeo1116.sakura.ne.jp/UTTT/UTTT.html (更新サボり中)

## ゲームエンジンの仕様について

UTTTGame/engine フォルダに格納  
学習に合わせて自由に改造できたほうがよいので、自前で実装した。モデルの学習はPytorchで行うので、接続しやすいようにゲームエンジンもPythonで実装した。  

### Game
通常のバトルをするclass  
2つのagentを対戦させて、その結果をRecordに記録する。  
Recordに記録されるのは
- player_index（1 or 2）
- 先手/後手
- エージェントの名前
- （着手前の）盤面の情報
- 合法手のリスト
- 指した手
- ゲーム自体の勝敗

ここで、ゲーム中にagentに与えられる盤面情報とRecordに記録される盤面情報はどちらもプレイヤーの主観情報となっており、空きマスが0、自分のマークが1、相手のマークが2で表されている。これは自分のplayer_indexが2であっても同じである（学習しやすくするため）

### ParallelGame
複数のゲームを同時に進行させるためのclass  
GPUを使ってポリシーネットワークで手を決める際には、複数のゲームについて個別に手を推論させるよりも1度にまとめたほうが全体として高速になる。強化学習やバリューネットワークの生成では自己対戦を含めてポリシーネットワーク同士の対戦を何千万回も行う必要があるので、こちらを使って対戦している。  

## 教師あり学習（ポリシーネットワーク）
学習の1st STEPとしてポリシーネットワーク教師あり学習をする。UTTTには人間同士の対極記録がない（多分）ので、MCTSエージェントの指し手を教師データとする。  
テスト実装ではResNetを使って指し手の一致率60％を達成した。

### 棋譜生成
1ターンに10000プレイアウトするMCTSエージェント（MctsAgent_10000）を使って対戦を行い、棋譜を集める。単純にMCTSエージェント同士を対戦させると試合の展開がワンパターンになってしまうと考えられるので、（RandomAgent, MctsAgent_1000, MctsAgent_5000, MctsAgent_10000）の4つのagentをランダムに用いるMixedAgentというエージェントを作って対戦を行った。  
1戦100秒程度と非常に時間がかかる。テストでは40スレッド × 5台 × 数日でMCTS1000の指し手を400万局面程度生成し、さらにそれを回転、反転することで3000万局面程度の教師データを作成した。
```
UTTTGame$ python3 -m learn.make_record --batch_size 200 --batch_num 30 --save_path ./learn/records/test --parallel_num 15
```

### ポリシーネットワークの学習

```
UTTTGame$ python3 -m learn.supervised_learning_policy --records_path ./learn/records --output_path ./learn/SL_output_test --epoch 50
```

## 教師あり学習（バリューネットワーク）
pass

### 教師データ生成
```
UTTTGame$ python3 -m learn.make_valuedata --batch_size 200 --policy_a ./policy_a.pth --policy_b ./policy_b.pth --batch_num 30 --save_path ./learn/records/test --parallel_num 128
```

### バリューネットワークの学習
```
UTTTGame$ python3 -m learn.supervised_learning_value --records_path ./learn/valuedata --output_path ./learn/SL_value_test --epoch 20
```

## 強化学習
2nd STEPとして強化学習をする。まだテスト中だが強くなってる気がしたので先んじてReDiveを始めておく
