#!/bin/sh

OUTPUT_PATH=$1
PROCESS_NUM=$2
PROCESS_NAME=$3

python3 -u -m learn.reinforcement.initialize --proc_name $PROCESS_NAME --output_path $OUTPUT_PATH --init_model ./learn/SL_0615_1000/models/state_97.pth
nohup python3 -u -m learn.reinforcement.learn --proc_name $PROCESS_NAME --input_path $OUTPUT_PATH --device "cuda:2" --lr 0.000001 >> $OUTPUT_PATH/logs/learn.txt < /dev/null &

for ((i=0; i<$PROCESS_NUM; i++)) do
    nohup python3 -u -m learn.reinforcement.battle --proc_name $PROCESS_NAME --output_path $OUTPUT_PATH >> $OUTPUT_PATH/logs/battle_$i.txt < /dev/null &
done

nohup python3 -u -m learn.reinforcement.evaluate --proc_name $PROCESS_NAME --input_path $OUTPUT_PATH --eval_model ./learn/SL_0615_1000/models/state_97.pth >> $OUTPUT_PATH/logs/evaluate.txt < /dev/null &
nohup python3 -u -m learn.reinforcement.evaluate --proc_name $PROCESS_NAME --input_path $OUTPUT_PATH --eval_model ./learn/SL_0608/models/state_625.pth >> $OUTPUT_PATH/logs/evaluate2.txt < /dev/null &