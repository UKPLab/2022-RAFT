#!/bin/bash
#SBATCH -J rational-hidden_states
#SBATCH -e /storage/ukp/work/fang/rational_bert/logs/finetune/%x.err
#SBATCH -o /storage/ukp/work/fang/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=50G 
#SBATCH --gres=gpu:1
#SBATCH --array 0-2,4-6

# change name, logging_steps,metric_for_best_model
activate_func="rational"
rational_layers="0-11,pooler"
lr=0
rational_lr=0
# seed=49

suffix=""

echo JOB ID: "${SLURM_JOBID}"
source /storage/ukp/work/fang/miniconda3/bin/activate /storage/ukp/work/fang/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="zsl"
export WANDB_ENTITY="haishuo"
export CUDA_LAUNCH_BLOCKING=1
# for rnd_seed in 20220422 20220423
# do

if [[ ${suffix} == "pretrained" ]]
then
  model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/acabert/rational/7e-4/12layers/adamw/0.005rlr-constlr-pooler
fi

if [ $SLURM_ARRAY_TASK_ID == 0 ]
then
  dataset_name=qnli
  if [[ ${suffix} == "100" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/qnli/rational/100/5e-5-1e-3-49-
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/qnli/rational/fulldata/5e-5-r1e-4-49-/checkpoint-7365
  fi
elif [ $SLURM_ARRAY_TASK_ID == 1 ]
then
  dataset_name=mrpc
  if [[ ${suffix} == "100" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mrpc/rational/100/5e-5-1e-3-49-
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mrpc/rational/fulldata/5e-5-r1e-4-49-/checkpoint-602
  fi
elif [ $SLURM_ARRAY_TASK_ID == 2 ]
then
  dataset_name=qqp

  if [[ ${suffix} == "100" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/qqp/rational/100/5e-5-5e-3-49-/checkpoint-28
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/qqp/rational/fulldata/5e-5-r5e-4-49-
  fi
elif [ $SLURM_ARRAY_TASK_ID == 3 ]
then
  dataset_name=mnli
  if [[ ${suffix} == "100" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/rational/100/5e-5-5e-3-49-
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/rational/fulldata/5e-5-r5e-4-49-
  fi
elif [ $SLURM_ARRAY_TASK_ID == 4 ]
then
  dataset_name=rte
  if [[ ${suffix} == "100" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/rational/fulldata/5e-5-r1e-3-20220602-100
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/rte/rational/fulldata/5e-5-r5e-4-49-
  fi
elif [ $SLURM_ARRAY_TASK_ID == 5 ]
then
  dataset_name=cola
  if [[ ${suffix} == "100" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/cola/rational/100/5e-5-5e-3-49-

  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/cola/rational/fulldata/5e-5-r5e-4-49-/checkpoint-1608
  fi

elif [ $SLURM_ARRAY_TASK_ID == 6 ]
then
  dataset_name=sst2
  if [[ ${suffix} == "100" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/sst2/rational/100/5e-5-1e-3-49-
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/sst2/rational/fulldata/5e-5-r1e-3-49-/checkpoint-14211
  fi
fi


python3 run_glue.py \
  --model_name_or_path ${model_name_or_path} \
  --model_type bert \
  --task_name ${dataset_name} \
  --cache_dir ./data_temp \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 5 \
  --output_dir ./outputs/probing_100000/$dataset_name/ \
  --save_strategy epoch \
  --logging_strategy epoch \
  --save_rational_plots \
  --greater_is_better True \
  --rational_lr ${rational_lr} \
  --report_to none \
  --name_suffix ${suffix}_${activate_func} \
  --overwrite_output_dir \
  --weight_decay 0.1 \
  --seed 49 \
  --academicBERT \
  # --max_train_samples ${suffix} \

  # --zero_shot \
# done
  # --dataset_name ${dataset_name} \