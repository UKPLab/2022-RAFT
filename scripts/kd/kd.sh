#!/bin/bash
#SBATCH -J rational-kd
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/kd/imp.err.%x
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/kd/imp.out.%x
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4

# change name, logging_steps,metric_for_best_model
activate_func="rational"
rational_layers="0-11"
lr=1e-4
config_name="config_12.json"
layers=$(echo ${config_name}| cut -d'.' -f 1| cut -d'_' -f 2)
optimizer="adam"


echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="kd"


# torchrun --nproc_per_node=1 \
torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --master_port 6061 \
    run_distillation.py \
    --model_name_or_path roberta-base \
    --model_type roberta \
    --dataset_name bookcorpus \
    --config_name ./outputs/roberta/${config_name} \
    --do_train \
    --max_seq_length 128 \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy epoch \
    --learning_rate $lr \
    --rational_layers $rational_layers \
    --num_train_epochs 3 \
    --eval_accumulation_steps 3200 \
    --report_to wandb \
    --logging_steps 2500 \
    --save_steps 2500 \
    --preprocessing_num_workers 8 \
    --output_dir ./outputs/roberta/kd/$lr/"${layers}layers"/$optimizer/ \
    --approx_func gelu \
    --run_name "${layers}-${config_name}-${optimizer}" \
    --weight_decay 0.01 \
    --alpha_ce 0.0 \
    --alpha_mlm 1.0 \
    --alpha_cos 0.0 \
    --alpha_clm 0.0 \
    --overwrite_output_dir \
