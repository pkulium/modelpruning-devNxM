#!/bin/bash

DEVICE=0

TASK_NAME=rte
CONFIG=../configs/uniform_fused_nxm_quantized_bert.json
MODEL=/scratch/admm_ds/v3/long_train/rte/7e-5_4e-3_16


CUDA_VISIBLE_DEVICES=$DEVICE python ../run_glue_admm.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL \
    --do_eval \
    --per_device_eval_batch_size=128 \
    --do_admm \
    --admm_config $CONFIG \
    --overwrite_output_dir \
    --output_dir $MODEL/eval_exp/per_sequence
