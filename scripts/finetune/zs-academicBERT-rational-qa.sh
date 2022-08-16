#!/bin/bash
#SBATCH -J rational-squad-em
#SBATCH -e /storage/ukp/work/fang/rational_bert/logs/finetune/%x.err
#SBATCH -o /storage/ukp/work/fang/rational_bert/logs/finetune/%x.out
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
source /storage/ukp/work/fang/miniconda3/bin/activate /storage/ukp/work/fang/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="squad"
export WANDB_ENTITY="haishuo"
# for rnd_seed in 20220422 20220423
# do

if [ $SLURM_ARRAY_TASK_ID == 0 ]
then
  seed=2022
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/full_data/1e-4-5e-4-2022-frozen_rf
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/300/1e-4-5e-4-2022-/checkpoint-50
  fi
elif [ $SLURM_ARRAY_TASK_ID == 1 ]
then
  seed=2259
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/full_data/1e-4-5e-4-2259-frozen_rf
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/300/1e-4-5e-4-2259-/checkpoint-120
  fi
elif [ $SLURM_ARRAY_TASK_ID == 2 ]
then
  seed=5309
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/full_data/1e-4-5e-4-5309-frozen_rf
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/300/1e-4-5e-4-5309-/checkpoint-130
  fi
elif [ $SLURM_ARRAY_TASK_ID == 3 ]
then
  seed=202206
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/full_data/1e-4-5e-4-202206-frozen_rf
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/300/1e-4-5e-4-202206-/checkpoint-100
  fi
elif [ $SLURM_ARRAY_TASK_ID == 4 ]
then
  seed=20220602
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/full_data/1e-4-5e-4-20220602-frozen_rf
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/300/1e-4-5e-4-20220602-/checkpoint-30
  fi
elif [ $SLURM_ARRAY_TASK_ID == 5 ]
then
  seed=49
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/full_data/1e-4-5e-4-49-frozen_rf
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/300/1e-4-5e-4-49-/checkpoint-40
  fi

elif [ $SLURM_ARRAY_TASK_ID == 6 ]
then
  seed=1046
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/full_data/1e-4-5e-4-49-frozen_rf
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/300/1e-4-1e-3-1046-frozen_rf/checkpoint-100
  fi
  
elif [ $SLURM_ARRAY_TASK_ID == 7 ]
then
  seed=0622
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/full_data/1e-4-5e-4-49-frozen_rf
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/300/1e-4-5e-4-0622-/checkpoint-90
  fi

elif [ $SLURM_ARRAY_TASK_ID == 8 ]
then
  seed=0320
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/full_data/1e-4-5e-4-49-frozen_rf
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/300/1e-4-5e-4-0320-/checkpoint-90
  fi

elif [ $SLURM_ARRAY_TASK_ID == 9 ]
then
  seed=53
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/full_data/1e-4-5e-4-49-frozen_rf
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/squad/rational/300/1e-4-5e-4-53-/checkpoint-110
  fi

fi


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
    # --cached_file ${dataset_name} \
      # --predict_file ./qa/web-dev-s.json \
      # --cached_file ${dataset_name} \