export TASK_NAME=rte
export MODEL=bert-base-uncased

DEVICE=1
BATCH_SIZES=("32")
LEARNING_RATES=("5e-5")

for LEARNING_RATE in ${LEARNING_RATES[*]}
do
    for BATCH_SIZE in ${BATCH_SIZES[*]}
    do
        CUDA_VISIBLE_DEVICES=$DEVICE python ../run_glue_admm.py \
            --task_name $TASK_NAME \
            --model_name_or_path $MODEL \
            --do_train \
            --do_eval \
            --per_device_eval_batch_size=128  \
            --per_device_train_batch_size=$BATCH_SIZE   \
            --gradient_accumulation_steps=1 \
            --learning_rate $LEARNING_RATE \
            --num_train_epochs 5.0 \
            --save_steps 30000 \
            --logging_steps 1000 \
            --overwrite_output_dir \
            --output_dir ./base_checkpoints/$TASK_NAME/
            # --output_dir ./scratch/admm_ds/v2/base_checkpoints/$TASK_NAME/
    done
    
done
