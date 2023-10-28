TASK_NAME=cola
MODEL=/scratch/admm_ds/v2/base_checkpoints/$TASK_NAME/
EXPERIMENT_NAME="exclude_10_epoch_random_sweep"

RANDOM_SEEDS=("1")

DEVICE=0
LEARNING_RATE="7e-5"
ADMM_RHO="4e-3"
BATCH_SIZE="16"
CONFIG="../configs/exclude_encoder/exclude_0.json"

for SEED in ${RANDOM_SEEDS[*]}
do
    CONFIG_INSTANCE=$(echo ${CONFIG} | awk -F "/" '{print $4}')
    CUDA_VISIBLE_DEVICES=$DEVICE python -m pdb ../run_glue_admm.py \
        --task_name $TASK_NAME \
        --model_name_or_path $MODEL \
        --do_train \
        --do_eval \
        --seed $SEED \
        --evaluation_strategy=epoch \
        --per_device_eval_batch_size=128  \
        --per_device_train_batch_size=$BATCH_SIZE \
        --gradient_accumulation_steps=1 \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs 10.0 \
        --save_steps 100000 \
        --logging_steps 2000 \
        --do_admm \
        --rho $ADMM_RHO \
        --admm_config $CONFIG \
        --overwrite_output_dir \
        --output_dir /scratch/admm_ds/v3/$EXPERIMENT_NAME/$TASK_NAME/$CONFIG_INSTANCE/$SEED/
done
