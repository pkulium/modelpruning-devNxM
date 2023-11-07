# TASK_NAMES=("stsb" "cola" "qnli" "sst2" "qqp" "mnli")
# EXPERIMENT_NAME="test_disabled"
# DEVICE=0
# LEARNING_RATES=("5e-5" "7e-5" "1e-4")
# ADMM_RHOS=("1e-3" "4e-3" "1e-2")
# BATCH_SIZES=("16")
# INT_QUANT=4
# CHUNK_SIZE=64
# CONFIG="../configs/admm_disabled/nxm_uniform.json"

# for TASK_NAME in ${TASK_NAMES[*]}
#     do
#     # MODEL=$HOME/$TASK_NAME/
#     MODEL=bert-base-uncased

#     for LEARNING_RATE in ${LEARNING_RATES[*]}
#     do
#         for BATCH_SIZE in ${BATCH_SIZES[*]}
#         do
#             CUDA_VISIBLE_DEVICES=$DEVICE python ../run_glue_admm.py \
#                 --task_name $TASK_NAME \
#                 --model_name_or_path $MODEL \
#                 --do_train \
#                 --do_eval \
#                 --evaluation_strategy=epoch \
#                 --per_device_eval_batch_size=128  \
#                 --per_device_train_batch_size=$BATCH_SIZE \
#                 --gradient_accumulation_steps=1 \
#                 --learning_rate $LEARNING_RATE \
#                 --num_train_epochs 10.0 \
#                 --save_steps 100000 \
#                 --logging_steps 2000 \
#                 --do_compression \
#                 --disable_admm \
#                 --admm_config ${CONFIG} \
#                 --overwrite_output_dir \
#                 --output_dir $HOME/output/$EXPERIMENT_NAME/$TASK_NAME/asp/${LEARNING_RATE}_${ADMM_RHO}_${BATCH_SIZE}/
#         done
#     done
# done

git pull

# cd ../../../
# /work/LAS/wzhang-lab/mingl/.conda/envs/fs/bin/python -m pip install  -e .
# cd experiments/transformers/scripts/

# TASK_NAMES=("stsb" "cola" "qnli" "sst2" "qqp" "mnli")
# LEARNING_RATES=("1e-4")
# ADMM_RHOS=("4e-3")

TASK_NAMES=("rte")
EXPERIMENT_NAME="nxm_bert"
DEVICE=0
BATCH_SIZE="32"
# LEARNING_RATES=("4e-4" "8e-4" "1e-4" "4e-5" "8e-5" "1e-5")
LEARNING_RATES=("1e-4")
# ADMM_RHOS=("1e-4" "1e-3" "1e-2" "1e-1")
ADMM_RHOS=("4e-3" "8e-3" "1e-3" "1e-2" "4e-2" "8e-2" "1e-1" "4e-1" "8e-1")
CONFIG="../configs/uniform_nxm_bert_lora.json"

for TASK_NAME in ${TASK_NAMES[*]}
    do
    # MODEL=$HOME/$TASK_NAME/
    MODEL="bert-base-cased"

    for LEARNING_RATE in ${LEARNING_RATES[*]}
    do
        for ADMM_RHO in ${ADMM_RHOS[*]}
        do
            CUDA_VISIBLE_DEVICES=$DEVICE python ../run_glue_admm.py \
                --task_name $TASK_NAME \
                --model_name_or_path $MODEL \
                --do_train \
                --do_eval \
                --evaluation_strategy=epoch \
                --per_device_eval_batch_size=128  \
                --per_device_train_batch_size=$BATCH_SIZE \
                --gradient_accumulation_steps=1 \
                --learning_rate $LEARNING_RATE \
                --num_train_epochs 25 \
                --do_compression \
                --save_steps 100000 \
                --logging_steps 2000 \
                --admm_config ${CONFIG} \
                --rho $ADMM_RHO \
                --overwrite_output_dir \
                --output_dir ./output_lora/$EXPERIMENT_NAME/$TASK_NAME/${LEARNING_RATE}_${ADMM_RHO}_${BATCH_SIZE}/
        done
    done
done

