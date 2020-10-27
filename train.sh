#!/usr/bin/env bash
set -eu

DATA_DIRPATH="./data"
OUTPUT_DIRPATH="result"

#CN_MODEL_PATH="PretrainCelebA/model_cn"
#tensorboard --logdir ./runs

python train.py \
        ${DATA_DIRPATH} \
        ${OUTPUT_DIRPATH} \
