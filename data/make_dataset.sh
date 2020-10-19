#!/usr/bin/env bash
set -eu

DATA_DIRPATH="./data"

#CN_MODEL_PATH="PretrainCelebA/model_cn"

python make_dataset.py \
        ${DATA_DIRPATH} \