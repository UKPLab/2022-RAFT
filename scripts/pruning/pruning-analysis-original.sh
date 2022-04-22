#!/bin/bash
#SBATCH -J pruning-analysis-original
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/pruning/imp.err.%x
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/pruning/imp.out.%x
#SBATCH -n 1
#SBATCH --mem=50G 
#SBATCH --gres=gpu:1

# change name, logging_steps,metric_for_best_model
activate_func='original'
rational_layers="''"
lr=5e-5
task_name="qnli"


echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/ukp-storage-1/fang/miniconda3/lib/"
module load cuda/11.1
# export WANDB_CONFIG_DIR="./wandb/"
# export WANDB_PROJECT="pruning"


# torchrun --nproc_per_node=1 \
python pruning_analysis.py \
  --model_name_or_path /ukp-storage-1/fang/pretrain_bert/outputs/qnli/original/pruning/checkpoint-39288 \
  --model_type roberta \
  --task_name $task_name \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 30 \
  --output_dir ./outputs/$task_name/$activate_func/pruning/ 
  # --save_rational_plots \
  # --overwrite_output_dir