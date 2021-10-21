# coding:utf-8

import multiprocessing
import os
import torch
import argparse
import matplotlib.pyplot as plt
from torch.nn.modules.activation import LeakyReLU
from learn.network.network import make_policynetwork, make_valuenetwork
from learn.util.reinforcementtrainer import ReinfocementTrainer


parser = argparse.ArgumentParser(description="自己対戦による強化学習")
parser.add_argument("--gpu_id", type=int, default=0, help="使用するgpuのid")
parser.add_argument("--temperature", type=float,
                    default=1.0, help="modelが手を選択するときの温度")
parser.add_argument("--policy_batchsize", type=int,
                    default=256, help="一度に並列対戦する回数（policy）")
parser.add_argument("--policy_batchnum", type=int, default=250,
                    help="1epochあたりに何batch対戦するか（policy）")
parser.add_argument("--value_batchsize", type=int,
                    default=8192, help="一度に並列対戦する回数（value）")
parser.add_argument("--value_batchnum", type=int, default=16,
                    help="1epochあたりに何batch対戦するか（value）")
parser.add_argument("--output_path", type=str,
                    default="./learn/RL_output", help="グラフ、モデルをoutputするディレクトリ")
parser.add_argument("--init_policy", type=str,
                    default=None, help="強化学習の初期ポリシー")
parser.add_argument("--init_value", type=str, default=None, help="初期value")
parser.add_argument("--policy_lr", type=float,
                    default=0.0000001, help="policyの基本学習率")
parser.add_argument("--value_lr", type=float,
                    default=0.0000001, help="valueの基本学習率")
args = parser.parse_args()

# outputするディレクトリの生成
os.makedirs(args.output_path, exist_ok=True)
os.makedirs(os.path.join(args.output_path, "models"), exist_ok=True)
os.makedirs(os.path.join(args.output_path, "graphs"), exist_ok=True)
# os.makedirs(os.path.join(args.output_path, "log"), exist_ok=True)

# モデルのロード
gpu_id = args.gpu_id
device = torch.device(f"cuda:{gpu_id}")
slpolicy = make_policynetwork()
slpolicy.to(device)
slpolicy = torch.nn.DataParallel(slpolicy, device_ids=[gpu_id])
learner = make_policynetwork()
learner.to(device)
learner = torch.nn.DataParallel(learner, device_ids=[gpu_id])
enemy = make_policynetwork()
enemy.to(device)
enemy = torch.nn.DataParallel(enemy, device_ids=[gpu_id])
enemy.eval()

if args.init_policy is not None:
    slpolicy.load_state_dict(torch.load(args.init_policy))
    learner.load_state_dict(torch.load(args.init_policy))
    enemy.load_state_dict(torch.load(args.init_policy))

value = make_valuenetwork()
value.to(device)
value = torch.nn.DataParallel(value, device_ids=[gpu_id])

if args.init_value is not None:
    value.load_state_dict(torch.load(args.init_value))

plt_epochs, plt_winrates = [], []
alpha = 0.0000001

# outputするディレクトリの作成
os.makedirs(args.output_path, exist_ok=True)
os.makedirs(os.path.join(args.output_path, "models"), exist_ok=True)
os.makedirs(os.path.join(args.output_path, "graphs"), exist_ok=True)
# os.makedirs(os.path.join(args.output_path, "log"), exist_ok=True)

cpu_num = min(16, multiprocessing.cpu_count())
trainer = ReinfocementTrainer(learner, enemy, slpolicy, value, cpu_num)

trainer.save_lerner()

for epoch in range(10000):
    print(f"epoch {epoch} start")

    learner.eval()
    enemy.eval()
    slpolicy.eval()
    value.eval()

    # policyの学習
    print("learn policy")
    policy_dataloader = trainer.get_policy_dataloader(
        args.policy_batchsize, args.policy_batchnum, args.temperature)
    trainer.train_policy(policy_dataloader, args.policy_lr)

    # valueの学習
    print("learn value")
    value_dataloader = trainer.get_value_dataloader(
        args.value_batchsize, args.value_batchnum, args.temperature)
    trainer.train_value(value_dataloader, args.value_lr)

    # 強さチェック
    print("check winrate")
    win_epoch, lose_epoch, draw_epoch = trainer.check_winrate(args.policy_batchsize)
    print(f"win = {win_epoch}, lose = {lose_epoch}, draw = {draw_epoch}")

    plt_epochs.append(epoch)
    plt_winrates.append(win_epoch / (win_epoch + lose_epoch + draw_epoch))

    if epoch % 10 == 9:
        # policyとvalueのセーブ
        policy_path = f"{args.output_path}/models/policy_{epoch + 1}.pth"
        value_path = f"{args.output_path}/models/value_{epoch + 1}.pth"
        torch.save(learner.state_dict(), policy_path)
        torch.save(value.state_dict(), value_path)

        # グラフ出力
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ln1 = ax1.plot(plt_epochs, plt_winrates, "C0", label="win rate")
        h1, l1 = ax1.get_legend_handles_labels()
        ax1.legend(h1, l1)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("win rate")
        fig.savefig(f"{args.output_path}/graphs/img_{epoch + 1}.png")

        # enemyのモデルを入れ替える
        state_dict = trainer.save_lerner()
        trainer.change_enemy()
