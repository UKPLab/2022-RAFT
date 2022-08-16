#!/bin/bash
#SBATCH -J rational-qqp
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/finetune/%x.err
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1

#2022,202206,20220602,2259,49,5309
# change name, logging_steps,metric_for_best_model
activate_func="rational"
rational_layers="0-11,pooler"
lr=5e-5
rational_lr=1e-4
task_name="qqp"
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
  --model_name_or_path /ukp-storage-1/fang/rational_bert/outputs/acabert/rational/7e-4/12layers/adamw/0.005rlr-constlr-pooler \
  --model_type bert \
  --task_name $task_name \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --curve linear linear linear \
  --lr_schedule step step step \
  --warmup_proportion_list 0 0 0 \
  --evaluation_strategy epoch \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 3 \
  --output_dir ./outputs/$task_name/$activate_func/${num_ins}/${lr}-r${rational_lr}-${seed} \
  --run_name "${task_name}-${activate_func}-${lr}-r${rational_lr}-${seed}-${num_ins}" \
  --save_strategy epoch \
  --logging_strategy epoch \
  --save_rational_plots \
  --rational_lr ${rational_lr} \
  --report_to wandb \
  --weight_decay 0.1 \
  --seed ${seed} \
  --academicBERT \
  --load_best_model_at_end \
  --metric_for_best_model eval_accuracy \
  --greater_is_better True \
  --overwrite_output_dir \
  --patience 2 \
  # --rational_weight_decay 0.0 \
  # --max_train_samples ${num_ins} \
  # --max_eval_samples ${num_ins} \


# done