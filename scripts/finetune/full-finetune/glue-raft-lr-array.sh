#!/bin/bash
#SBATCH -J lr-rational-stsb-100
#SBATCH -e xxx/rational_bert/logs/finetune/%x.err
#SBATCH -o xxx/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9

#2022,202206,20220602,2259,49,5309
# change name, logging_steps,metric_for_best_model
activate_func="rational"
rational_layers="0-11,pooler"
lr=1e-2
rational_lr=5e-4
task_name="stsb"
num_ins=100
suffix="ch"

if [ $SLURM_ARRAY_TASK_ID == 0 ]
then
  seed=2022
elif [ $SLURM_ARRAY_TASK_ID == 1 ]
then
  seed=202206
elif [ $SLURM_ARRAY_TASK_ID == 2 ]
then
  seed=20220602
elif [ $SLURM_ARRAY_TASK_ID == 3 ]
then
  seed=2259
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
source xxx/miniconda3/bin/activate xxx/miniconda3
module load cuda/11.1
export WANDB_DIR="xxx/rational_bert/wandb_t"
export WANDB_CONFIG_DIR="xxx/rational_bert/wandb_t"
export WANDB_PROJECT="glue-deepnet-lr"

# for rnd_seed in 20220422 20220423
# do

python3 run_glue.py \
  --model_name_or_path xxx/rational_bert/outputs/acabert/rational/7e-4/12layers/adamw/0.005rlr-constlr-pooler \
  --model_type bert \
  --cache_dir ./data_temp \
  --task_name $task_name \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --curve linear linear linear \
  --lr_schedule step step step \
  --num_warmup_steps 0 0 0 \
  --evaluation_strategy epoch \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 30 \
  --output_dir ./outputs/$task_name/$activate_func/${num_ins}/${lr}-${rational_lr}-${seed}-${suffix} \
  --run_name "${task_name}-${activate_func}-${lr}-r${rational_lr}-${seed}-${num_ins}-${suffix}" \
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_strategy epoch \
  --save_rational_plots \
  --rational_lr ${rational_lr} \
  --report_to wandb \
  --weight_decay 0.1 \
  --seed ${seed} \
  --academicBERT \
  --load_best_model_at_end \
  --metric_for_best_model eval_combined_score \
  --greater_is_better True \
  --overwrite_output_dir \
  --max_train_samples ${num_ins} \
  --max_eval_samples ${num_ins} \
  --patience 20 \
  --frozen_model \