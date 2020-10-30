#!/usr/bin/env bash
set -eu

DATA_DIRPATH="./result/prediction"
OUTPUT_DIRPATH="./result/prediction/eval"
MODEL_DIRPATH="./result/train_ckpt"


#CN_MODEL_PATH="PretrainCelebA/model_cn"
#tensorboard --logdir ./runs

python eval.py \
        ${DATA_DIRPATH} \
        ${OUTPUT_DIRPATH} \
        --model=${MODEL_DIRPATH}