#!/bin/sh

OUTPUT_PATH=$1
PROCESS_NUM=$2
PROCESS_NAME=$3

for ((i=0; i<$PROCESS_NUM; i++)) do
    nohup python3 -u -m learn.reinforcement.battle --proc_name $PROCESS_NAME --output_path $OUTPUT_PATH --temperature 0.01 &
    # nohup python3 -u -m learn.reinforcement.battle --proc_name $PROCESS_NAME --output_path $OUTPUT_PATH --temperature 0.01 >> $OUTPUT_PATH/logs/battle_b_$i.txt < /dev/null &
done

sleep 1000000