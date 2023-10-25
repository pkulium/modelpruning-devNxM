TASK_NAME=rte
MODEL=/scratch/admm_ds/v2/base_checkpoints/$TASK_NAME/
EXPERIMENT_NAME="long_train"

DEVICE=0
LEARNING_RATES=("5e-5" "7e-5" "9e-5")
ADMM_RHOS=("1e-3" "4e-3" "1e-2")
BATCH_SIZES=("16")

CONFIGS=(
    ../configs/uniform_fused_nxm_quantized_bert.json
)

# Number of epochs to train is inclusive of the number of epochs originally used
# for training. In this case, that means to train 10 epochs we supply 15 as an argument

for CONFIG in ${CONFIGS[*]}
do
    for LEARNING_RATE in ${LEARNING_RATES[*]}
    do
        for ADMM_RHO in ${ADMM_RHOS[*]}
        do
            for BATCH_SIZE in ${BATCH_SIZES[*]}
            do
                CONFIG_INSTANCE=$(echo ${CONFIG} | awk -F "/" '{print $4}')
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
                    --num_train_epochs 15.0 \
                    --save_steps 100000 \
                    --logging_steps 2000 \
                    --do_admm \
                    --rho $ADMM_RHO \
                    --admm_config $CONFIG \
                    --overwrite_output_dir \
                    --output_dir /scratch/admm_ds/v3/$EXPERIMENT_NAME/$TASK_NAME/$CONFIG_INSTANCE/${LEARNING_RATE}_${ADMM_RHO}_${BATCH_SIZE}/
            done
        done
    done
done