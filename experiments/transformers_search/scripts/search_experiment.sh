EXPERIMENT_DIR="../output"
EXPERIMENT_CONFIG="../configs/base_config.json"
TOP_K=5
TASK_NAME="mrpc"
DEVICE=0
MODEL="/scratch/admm_ds/v2/base_checkpoints/${TASK_NAME}"

CUDA_VISIBLE_DEVICES=$DEVICE python ../search_experiment_candidates.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL \
    --experiment $EXPERIMENT_CONFIG \
    --experiment_dir $EXPERIMENT_DIR \
    --top_k $TOP_K
