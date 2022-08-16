#!/bin/bash
#SBATCH -J original-hidden_states
#SBATCH -e /storage/ukp/work/fang/rational_bert/logs/finetune/%x.err
#SBATCH -o /storage/ukp/work/fang/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=50G 
#SBATCH --gres=gpu:1
#SBATCH --array 0-6

# change name, logging_steps,metric_for_best_model
activate_func="original"
rational_layers="''"
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
  model_name_or_path=/ukp-storage-1/fang/rational_bert/outputs/acabert/original/7e-4/12layers/adamw/deepnet-0.01warm
fi



if [ $SLURM_ARRAY_TASK_ID == 0 ]
then
  dataset_name=qnli
  if [[ ${suffix} == "100" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/qnli/original/100/5e-5-49/checkpoint-72
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/qnli/original/fulldata/5e-5-49/checkpoint-7365
  fi
elif [ $SLURM_ARRAY_TASK_ID == 1 ]
then
  dataset_name=mrpc
  if [[ ${suffix} == "100" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mrpc/original/100/2e-5-49/checkpoint-72
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mrpc/original/fulldata/5e-5-49/checkpoint-774
  fi
elif [ $SLURM_ARRAY_TASK_ID == 2 ]
then
  dataset_name=qqp

  if [[ ${suffix} == "100" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/qqp/original/100/5e-5-49/checkpoint-36
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/qqp/original/fulldata/5e-5-49
  fi
elif [ $SLURM_ARRAY_TASK_ID == 3 ]
then
  dataset_name=mnli
  if [[ ${suffix} == "100" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/original/100/2e-5-49/checkpoint-76
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/original/fulldata/5e-5-49
  fi
elif [ $SLURM_ARRAY_TASK_ID == 4 ]
then
  if [[ ${suffix} == "100" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/rational/fulldata/5e-5-r1e-3-20220602-frozen
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/rte/original/fulldata/5e-5-49
  fi
elif [ $SLURM_ARRAY_TASK_ID == 5 ]
then
  dataset_name=cola
  if [[ ${suffix} == "100" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/cola/original/100/5e-5-49
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/cola/original/fulldata/5e-5-49/checkpoint-1407
  fi
elif [ $SLURM_ARRAY_TASK_ID == 6 ]
then
  dataset_name=sst2
  if [[ ${suffix} == "100" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/sst2/original/100/5e-5-49/checkpoint-56
  elif [[ ${suffix} == "pretrained" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/acabert/original/7e-4/12layers/adamw/deepnet-0.01warm
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/sst2/original/fulldata/5e-5-49/checkpoint-14211
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
  --name_suffix ${suffix}_${activate_func}  \
  --report_to none \
  --overwrite_output_dir \
  --weight_decay 0.1 \
  --seed 49 \
  --academicBERT \
    # --max_train_samples ${suffix} \
  # --zero_shot \
    # --dataset_name ${dataset_name} \
# done