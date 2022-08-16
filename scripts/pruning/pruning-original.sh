#!/bin/bash
#SBATCH -J original-mrpc
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/pruning/%x.err
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/pruning/%x.out
#SBATCH -n 1
#SBATCH --mem=50G 
#SBATCH --gres=gpu:1

# change name, logging_steps,metric_for_best_model
activate_func="original"
rational_layers="''"
lr=5e-5
task_name="mrpc"


echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="pruning"
export WANDB_ENTITY="haishuo"

# torchrun --nproc_per_node=1 \
python run_glue.py \
  --model_name_or_path /ukp-storage-1/fang/rational_bert/outputs/acabert/original/7e-4/12layers/adamw/postln \
  --model_type bert \
  --task_name $task_name \
  --do_train \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --evaluation_strategy epoch \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 30 \
  --output_dir ./outputs/$task_name/$activate_func/pruning/ \
  --run_name "academicbert-${task_name}-${activate_func}" \
  --save_strategy epoch \
  --logging_strategy epoch \
  --logging_steps 268 \
  --save_steps 804 \
  --rational_lr 0.01 \
  --do_pruning \
  --academicBERT \
  --overwrite_output_dir
  # --save_rational_plots \
 