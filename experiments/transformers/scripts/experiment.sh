TASK_NAMES=("stsb" "cola" "qnli" "sst2" "qqp" "mnli")
EXPERIMENT_NAME="quant_with_activ"
DEVICE=0
LEARNING_RATES=("5e-5" "7e-5" "1e-4")
ADMM_RHOS=("1e-3" "4e-3" "1e-2")
BATCH_SIZES=("16")
INT_QUANT=4
CHUNK_SIZE=16
CONFIG="../configs/quantization/modified_${INT_QUANT}_bit_${CHUNK_SIZE}_chunk.json"

for TASK_NAME in ${TASK_NAMES[*]}
    do
    MODEL=$HOME/$TASK_NAME/

    for LEARNING_RATE in ${LEARNING_RATES[*]}
    do
        for ADMM_RHO in ${ADMM_RHOS[*]}
        do
            for BATCH_SIZE in ${BATCH_SIZES[*]}
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
                    --num_train_epochs 10.0 \
                    --save_steps 100000 \
                    --logging_steps 2000 \
                    --do_compression \
                    --rho $ADMM_RHO \
                    --admm_config ${CONFIG} \
                    --overwrite_output_dir \
                    --output_dir $HOME/output/$EXPERIMENT_NAME/$TASK_NAME/modified_${INT_QUANT}_${CHUNK_SIZE}/${LEARNING_RATE}_${ADMM_RHO}_${BATCH_SIZE}/
            done
        done
    done
done
