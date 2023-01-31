#!/bin/bash
#SBATCH -J rational-squad-lr-100
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/finetune/%x.err
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu 3
#SBATCH --array=0-8

#2022,202206,20220602,2259,49,5309
# change name, logging_steps,metric_for_best_model
activate_func="rational"
rational_layers="0-11"
lr=1e-4
rational_lr=1e-4
dataset_name="squad"
num_ins=100
suffix="128seq_fixed"

if [ $SLURM_ARRAY_TASK_ID == 0 ]
then
  seed=2022
elif [ $SLURM_ARRAY_TASK_ID == 1 ]
then
  seed=202206
elif [ $SLURM_ARRAY_TASK_ID == 2 ]
then
  seed=20220602
  # seed=1708
elif [ $SLURM_ARRAY_TASK_ID == 3 ]
then
  seed=2259
  # seed=10341
elif [ $SLURM_ARRAY_TASK_ID == 4 ]
then
  seed=5309
elif [ $SLURM_ARRAY_TASK_ID == 5 ]
then
  seed=1046

elif [ $SLURM_ARRAY_TASK_ID == 6 ]
then
  seed=0622

elif [ $SLURM_ARRAY_TASK_ID == 7 ]
then
  seed=0320

elif [ $SLURM_ARRAY_TASK_ID == 8 ]
then
  seed=53

elif [ $SLURM_ARRAY_TASK_ID == 9 ]
then
  seed=49
fi

echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate xxx/miniconda3
module load cuda/11.1
export WANDB_DIR="./wandb"
export WANDB_CONFIG_DIR="/ukp-storage-1/fang/rational_bert/wandb"
export WANDB_PROJECT="squad"

python run_squad.py \
  --model_name_or_path /ukp-storage-1/fang/rational_bert/outputs/acabert/rational/7e-4/12layers/adamw/0.005rlr-constlr-pooler \
  --model_type bert \
  --do_train \
  --do_eval \
  --run_name squad_lr_${lr}_r_${rational_lr}_${seed}_${num_ins}_${suffix} \
  --cache_dir ./data_temp/squad \
  --data_dir ./data_temp/squad \
  --max_seq_length 128 \
  --evaluate_during_training \
  --per_gpu_train_batch_size 32 \
  --curve linear linear linear \
  --per_gpu_eval_batch_size 32 \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 20 \
  --output_dir ./outputs/$dataset_name/$activate_func/$num_ins/${lr}-${rational_lr}-${seed}-${suffix}/ \
  --rational_lr ${rational_lr} \
  --overwrite_output_dir \
  --weight_decay 0.1 \
  --seed ${seed} \
  --academicBERT \
  --logging_steps 0 \
  --save_steps 0 \
  --save_rational_plot \
  --max_train_samples ${num_ins} \
  --max_dev_samples ${num_ins} \
  --frozen_rf
