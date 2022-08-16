#!/bin/bash
#SBATCH -J original-squad
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/finetune/%x.err
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=50G 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu 3

# change name, logging_steps,metric_for_best_model
#2022,202206,20220602,2259,49,5309
activate_func="original"
rational_layers="''"
lr=2e-4
rational_lr=1e-4
dataset_name="squad"
seed=2259


echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="glue-academicBERT"
export WANDB_ENTITY="haishuo"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/ukp-storage-1/fang/miniconda3/lib/"
# for rnd_seed in 20220422 20220423
# do
python run_squad.py \
  --model_name_or_path /ukp-storage-1/fang/rational_bert/outputs/acabert/original/7e-4/12layers/adamw/deepnet \
  --model_type bert \
  --do_train \
  --do_eval \
  --cache_dir ./data/squad \
  --data_dir ./data/squad \
  --max_seq_length 384 \
  --evaluate_during_training \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 8 \
  --output_dir ./outputs/$dataset_name/$activate_func/${lr}-${seed}/ \
  --rational_lr ${rational_lr} \
  --overwrite_output_dir \
  --weight_decay 0.01 \
  --seed ${seed} \
  --academicBERT \
  --run_name "squad_lr_${lr}_origin_${seed}" \
  --max_train_samples 1000 \
  --max_dev_samples 1000 \
  --logging_steps 33 \
  --save_steps 33 \
  # 

  # --report_to

    # --model_name_or_path /ukp-storage-1/fang/rational_bert/outputs/acabert/original/7e-4/12layers/adamw/postln \

# done