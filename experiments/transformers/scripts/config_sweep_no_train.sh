#!/bin/bash

DEVICE=0

TASK_NAMES=(
    mnli
)
CONFIGS=(
    ../configs/exclude_encoder/exclude_0.json
    ../configs/exclude_encoder/exclude_1.json
    ../configs/exclude_encoder/exclude_2.json
    ../configs/exclude_encoder/exclude_3.json
    ../configs/exclude_encoder/exclude_4.json
    ../configs/exclude_encoder/exclude_5.json
    ../configs/exclude_encoder/exclude_6.json
    ../configs/exclude_encoder/exclude_7.json
    ../configs/exclude_encoder/exclude_8.json
    ../configs/exclude_encoder/exclude_9.json
    ../configs/exclude_encoder/exclude_10.json
    ../configs/exclude_encoder/exclude_11.json
)
DEVICE=1

for TASK in ${TASK_NAMES[*]}
do
    for CONFIG in ${CONFIGS[*]}
    do
        CUDA_VISIBLE_DEVICES=$DEVICE python ../run_glue_admm.py \
            --task_name $TASK \
            --model_name_or_path /scratch/admm_ds/v2/base_checkpoints/$TASK/ \
            --do_eval \
            --per_device_eval_batch_size=128 \
            --do_admm \
            --admm_config $CONFIG \
            --overwrite_output_dir \
            --output_dir /scratch/admm_ds/v3/structured_sweep/$TASK/$CONFIG/
    done
done