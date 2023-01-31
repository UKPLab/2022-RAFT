#!/bin/bash
#SBATCH -J original-rte-bitfit
#SBATCH -e /storage/ukp/work/fang/rational_bert/logs/finetune/%x.err
#SBATCH -o /storage/ukp/work/fang/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=50G 
#SBATCH --gres=gpu:1
#SBATCH --array=1-4

#2022,202206,20220602,2259,49,5309
# change name, logging_steps,metric_for_best_model
activate_func="original"
rational_layers="''"
lr=1e-2
rational_lr=0
task_name="rte"
num_ins=100
suffix="sub_bias"

echo  $SLURM_ARRAY_TASK_ID
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
  seed=49
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
  seed=1046

fi

echo JOB ID: "${SLURM_JOBID}"
source /storage/ukp/work/fang/miniconda3/bin/activate /storage/ukp/work/fang/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="glue-deepnet-lr"


python3 run_glue.py \
  --model_name_or_path /storage/ukp/work/fang/rational_bert/outputs/acabert/original/7e-4/12layers/adamw/deepnet-0.01warm \
  --model_type bert \
  --task_name $task_name \
  --cache_dir ./data_temp \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --evaluation_strategy epoch \
  --curve linear linear linear \
  --lr_schedule step step step \
  --warmup_proportion_list 0.0 0.0 0.0 \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 30 \
  --output_dir ./outputs/$task_name/$activate_func/${num_ins}/${lr}-${seed}-${suffix} \
  --run_name "${task_name}-${activate_func}-${lr}-rn/a-${seed}-${num_ins}-${suffix}" \
  --save_strategy epoch \
  --logging_strategy epoch \
  --save_total_limit 2 \
  --load_best_model_at_end \
  --metric_for_best_model eval_accuracy \
  --greater_is_better True \
  --rational_lr ${rational_lr} \
  --report_to wandb \
  --weight_decay 0.1 \
  --seed ${seed} \
  --academicBERT \
  --overwrite_output_dir \
  --patience 20 \
  --max_train_samples ${num_ins} \
  --max_eval_samples ${num_ins} \
  --sub_bias \

  