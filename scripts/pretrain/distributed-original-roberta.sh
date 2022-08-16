#!/bin/bash
#SBATCH -J original-12layers-acdemicbert
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/pretrain/imp.err.%x
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/pretrain/imp.out.%x
#SBATCH -n 1
#SBATCH --mem=50G 
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu 3


activate_func="original"
rational_layers="''"
lr=6e-4
config_name="config_12.json"
layers=$(echo ${config_name}| cut -d'.' -f 1| cut -d'_' -f 2)
optimizer="adam"

# export MASTER_PORT=12340
# export WORLD_SIZE=4
# echo "NODELIST="${SLURM_NODELIST}
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR
# export NCCL_IB_DISABLE=1

# change name, config file, batch size, graident steps,  logging_steps, 

echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
# module load cuda/11.3
export WANDB_CONFIG_DIR="/ukp-storage-1/fang/rational_bert/wandb/"
export WANDB_PROJECT="Pretraining"
export WANDB_ENTITY="haishuo"

torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --master_port 6061 \
    run_mlm.py \
    --model_type roberta \
    --dataset_name wikipedia \
    --dataset_config_name 20200501.en \
    --config_name ./configs/${config_name} \
    --tokenizer_name roberta-base \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 16 \
    --do_train \
    --do_eval \
    --num_train_epochs 10 \
    --eval_accumulation_steps 3200 \
    --report_to wandb \
    --logging_steps 2500 \
    --save_steps 2500 \
    --max_seq_length 128 \
    --learning_rate $lr \
    --preprocessing_num_workers 8 \
    --output_dir ./outputs/roberta/$activate_func/$lr/"${layers}layers"/$optimizer/original_setup \
    --rational_layers $rational_layers \
    --approx_func gelu \
    --run_name "${activate_func}-${config_name}-${optimizer}" \
    --weight_decay 0.01 \
    --warmup_ratio 0.06 \
    --overwrite_output_dir \
    # --tb_dir ./runs/graident_check-original \

    # --add_ln True
    # --config_overrides ../output_models/roberta/roberta_config.json \