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


# TASK_NAMES=("stsb" "cola" "qnli" "sst2" "qqp" "mnli")
TASK_NAMES=("sst2" )
EXPERIMENT_NAME="nxm_bert"
DEVICE=0
LEARNING_RATES=("5e-5")
ADMM_RHOS=("3e-3")
BATCH_SIZES=("16")
CONFIG="../configs/uniform_nxm_bert.json"

for TASK_NAME in ${TASK_NAMES[*]}
    do
    # MODEL=$HOME/$TASK_NAME/
    MODEL="/work/LAS/wzhang-lab/mingl/code/lora/modelpruning-devNxM/experiments/transformers/scripts/output/test_disabled/stsb/5e-5__16"

    for LEARNING_RATE in ${LEARNING_RATES[*]}
    do
        for BATCH_SIZE in ${BATCH_SIZES[*]}
        do
            CUDA_VISIBLE_DEVICES=$DEVICE python -m pdb ../run_glue_admm.py \
                --task_name $TASK_NAME \
                --model_name_or_path $MODEL \
                --do_train \
                --do_eval \
                --evaluation_strategy=epoch \
                --per_device_eval_batch_size=128  \
                --per_device_train_batch_size=$BATCH_SIZE \
                --gradient_accumulation_steps=1 \
                --learning_rate $LEARNING_RATE \
                --num_train_epochs 5 \
                --save_steps 100000 \
                --logging_steps 2000 \
                --admm_config ${CONFIG} \
                --overwrite_output_dir \
                --output_dir ./output/$EXPERIMENT_NAME/$TASK_NAME/asp/${LEARNING_RATE}_${ADMM_RHO}_${BATCH_SIZE}/
        done
    done
done
