#!/bin/bash

ALG=$1
GAMMA=$2
EPSILON=$3
REWARD_SCALE=$4
OUTPUT_DIR=$5
TRAIN_PATHS=$6
VALID_PATH=$7
NGROUPS=$8

python run_clm.py \
    --model_name_or_path="" \
    --model_type=gpt2 \
    --tokenizer_name=gpt2 \
    --config_name=gpt2-6 \
    --train_file="$TRAIN_PATHS" \
    --validation_file="$VALID_PATH" \
    --do_train \
    --do_eval \
    --output_dir="$OUTPUT_DIR" \
    --cache_dir=../.cache \
    --num_groups="$NGROUPS" \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --eval_steps=1000 \
    --evaluation_strategy=steps \
    --scoring_function="$ALG" \
    --fp16 \
    --logging_steps=100 \
    --eval_steps=1000 \
    --save_steps=5000 \
    --overwrite_output_dir \
    --block_size=256 \
    --max_steps=100000 \
    --learning_rate=1e-4 \
    --warmup_steps=10000 \
    --num_eval_batches_for_reward=10 \
    --steps_per_reward=100 \
    --gamma="$GAMMA" \
    --reward_scale="$REWARD_SCALE" \
    --epsilon="$EPSILON"