#!/bin/bash
#SBATCH -J original-trivia
#SBATCH -e xxx/rational_bert/logs/finetune/%x.err
#SBATCH -o xxx/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=120G 
#SBATCH --gres=gpu:1
#SBATCH --array=0-5

# change name, logging_steps,metric_for_best_model
activate_func="original"
rational_layers="''"
lr=2e-5
rational_lr=1e-4
dataset_name=verified_cached_predict_384_web
num_ins=full_data

echo JOB ID: "${SLURM_JOBID}"
source xxx/miniconda3/bin/activate xxx/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="zsl"
export CUDA_LAUNCH_BLOCKING=1
# for rnd_seed in 20220422 20220423
# do


python run_squad.py \
  --model_name_or_path ${model_name_or_path} \
  --model_type bert \
  --do_eval \
  --run_name ${dataset_name}_${activate_func}_${seed}_${num_ins} \
  --cache_dir ./data_temp/triviaqa \
  --data_dir ./data_temp/triviaqa \
  --cached_file ${dataset_name} \
  --predict_file ./qa/verified-web-dev-s.json \
  --max_seq_length 384 \
  --evaluate_during_training \
  --per_gpu_train_batch_size 32 \
  --curve linear linear linear \
  --per_gpu_eval_batch_size 32 \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 3 \
  --output_dir xxx/rational_bert/outputs/squad/rational/full_data/1e-4-1e-3-49-frozen_rf \
  --rational_lr ${rational_lr} \
  --overwrite_output_dir \
  --weight_decay 0.1 \
  --seed ${seed} \
  --academicBERT \
  --logging_steps 2089 \


# done