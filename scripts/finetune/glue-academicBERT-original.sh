#!/bin/bash
#SBATCH -J original-rte
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/finetune/%x.err
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=50G 
#SBATCH --gres=gpu:1

#2022,202206,20220602,2259,49,5309
# change name, logging_steps,metric_for_best_model
activate_func="original"
rational_layers="''"
lr=5e-5
rational_lr=1e-4
task_name="rte"
seed=49
num_ins=fulldata


echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="glue-deepnet-ht-new"
export WANDB_ENTITY="haishuo"

# for rnd_seed in 20220422 20220423
# do

python3 run_glue.py \
  --model_name_or_path /ukp-storage-1/fang/rational_bert/outputs/acabert/original/7e-4/12layers/adamw/deepnet-0.01warm \
  --model_type bert \
  --task_name $task_name \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --evaluation_strategy epoch \
  --curve linear linear linear \
  --lr_schedule step step step \
  --num_warmup_steps 0 0 0 \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 3 \
  --output_dir ./outputs/$task_name/$activate_func/${num_ins}/${lr}-${seed} \
  --run_name "${task_name}-${activate_func}-${lr}-rn/a-${seed}-${num_ins}" \
  --save_strategy epoch \
  --logging_strategy epoch \
  --load_best_model_at_end \
  --metric_for_best_model eval_accuracy \
  --greater_is_better True \
  --rational_lr ${rational_lr} \
  --report_to wandb \
  --weight_decay 0.1 \
  --seed ${seed} \
  --academicBERT \
  --overwrite_output_dir \
  --patience 2 \
  #   --max_train_samples ${num_ins} \
  # --max_eval_samples ${num_ins} \
# done