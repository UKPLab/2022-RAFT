#!/bin/bash
#SBATCH -J rational-rte-hidden_states
#SBATCH -e xxx/rational_bert/logs/finetune/%x.err
#SBATCH -o xxx/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=50G 
#SBATCH --gres=gpu:1
#SBATCH --array 4

# change name, logging_steps,metric_for_best_model
activate_func="rational"
rational_layers="0-11,pooler"
lr=0
rational_lr=0
dataset_name=rte
model_name_or_path=your_model
# suffix="12"

echo JOB ID: "${SLURM_JOBID}"
source xxx/miniconda3/bin/activate xxx/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="zsl"
export CUDA_LAUNCH_BLOCKING=1




python3 run_glue.py \
  --model_name_or_path ${model_name_or_path} \
  --model_type bert \
  --task_name ${dataset_name} \
  --cache_dir ./data_temp \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 5 \
  --output_dir ./outputs/$dataset_name/$activate_func/ \
  --run_name "${dataset_name}-${activate_func}-${seed}-${suffix}" \
  --save_strategy epoch \
  --logging_strategy epoch \
  --save_rational_plots \
  --greater_is_better True \
  --rational_lr ${rational_lr} \
  --report_to none \
  --max_train_samples 1 \
  --overwrite_output_dir \
  --weight_decay 0.1 \
  --seed ${seed} \
  --academicBERT \
  --zero_shot \
