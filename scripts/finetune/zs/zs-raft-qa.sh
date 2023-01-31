#!/bin/bash
#SBATCH -J rational-squad-em
#SBATCH -e xxx/rational_bert/logs/finetune/%x.err
#SBATCH -o xxx/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=120G 
#SBATCH --gres=gpu:1
#SBATCH --array 6-9

# change name, logging_steps,metric_for_best_model
activate_func="rational"
rational_layers="0-11"
lr=1e-4
rational_lr=5e-4
dataset_name=squad
num_ins=300
suffix="em"

echo JOB ID: "${SLURM_JOBID}"
source xxx/miniconda3/bin/activate xxx/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="squad"




python run_squad.py \
  --model_name_or_path ${model_name_or_path} \
  --model_type bert \
  --do_eval \
  --run_name ${dataset_name}_${activate_func}_${seed}_${num_ins}_${suffix} \
  --cache_dir ./data_temp/squad \
  --data_dir ./data_temp/squad \
  --max_seq_length 384 \
  --evaluate_during_training \
  --per_gpu_train_batch_size 32 \
  --curve linear linear linear \
  --per_gpu_eval_batch_size 32 \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 3 \
  --output_dir ${model_name_or_path} \
  --rational_lr ${rational_lr} \
  --overwrite_output_dir \
  --weight_decay 0.1 \
  --seed ${seed} \
  --academicBERT \
  --logging_steps 2089 \
  --save_steps 2089 \
  --save_rational_plot \
