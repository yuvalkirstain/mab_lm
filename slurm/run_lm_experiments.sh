#!/bin/bash

OUTPUT_DIR="results"
DATA_DIR="data/raw_data/wikitext-103"
epsilon=-1
gamma=-1
REWARD_SCALE=10

for domain in "film" "road" "human"; do
  for alg in "exp3" "in_domain" "general" "epsilon_greedy"; do

    cur_output_dir="$OUTPUT_DIR/$alg-$domain"
    mkdir -p $cur_output_dir

    valid_path="$DATA_DIR/$domain/wiki-$domain-in-domain.valid.tokens"

    if [ $alg == "exp3" ]; then
      gamma="0.01"
      train_paths="$DATA_DIR/$domain/wiki*train.tokens"
    fi
    if [ $alg == "epsilon_greedy" ]; then
      epsilon="0.1"
      gamma="0.1"
      train_paths="$DATA_DIR/$domain/wiki*train.tokens"
    fi
    if [ $alg == "in_domain" ]; then
      train_paths="$DATA_DIR/$domain/wiki-$domain-in-domain.train.tokens"
    fi
    if [ $alg == "general" ]; then
      train_paths="$DATA_DIR/$domain/wiki.train.tokens"
    fi

    sbatch --job-name=mab_lm \
      --output="$cur_output_dir.out" \
      --error="$cur_output_dir.err" \
      --partition="killable"	\
      --time=1440 \
      --signal=USR1@120 \
      --nodes=1 \
      --ntasks=1 \
      --mem=50000 \
      --cpus-per-task=4 \
      --gpus-per-task=1 \
      slurm/run_single_lm_experiment.sh "$alg" "$gamma" "$epsilon" "$REWARD_SCALE" "$cur_output_dir" "$train_paths" "$valid_path"
  done
done
