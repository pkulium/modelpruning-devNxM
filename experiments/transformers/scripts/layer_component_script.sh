TASK_NAME=cola
MODEL=/scratch/admm_ds/v2/base_checkpoints/$TASK_NAME/
EXPERIMENT_NAME="layer_component_beta"

DEVICE=0
CUDA_VISIBLE_DEVICES=$DEVICE python ../layer_component_exploration.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL \
    --per_device_eval_batch_size=1  \
    --overwrite_output_dir \
    --output_dir /scratch/admm_ds/v3/$EXPERIMENT_NAME/


