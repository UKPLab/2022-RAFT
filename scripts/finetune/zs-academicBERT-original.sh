#!/bin/bash
#SBATCH -J original-rte-zs
#SBATCH -e /storage/ukp/work/fang/rational_bert/logs/finetune/%x.err
#SBATCH -o /storage/ukp/work/fang/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=50G 
#SBATCH --gres=gpu:1
#SBATCH --array 4

# change name, logging_steps,metric_for_best_model
activate_func="original"
rational_layers="''"
lr=0
rational_lr=0
# seed=49
dataset_name=rte
suffix="rte"

echo JOB ID: "${SLURM_JOBID}"
source /storage/ukp/work/fang/miniconda3/bin/activate /storage/ukp/work/fang/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="zsl"
export WANDB_ENTITY="haishuo"
export CUDA_LAUNCH_BLOCKING=1
# for rnd_seed in 20220422 20220423
# do

if [ $SLURM_ARRAY_TASK_ID == 0 ]
then
  seed=2022
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/original/fulldata/5e-5-r1e-3-2022-frozen
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/original/fulldata/5e-5-2022
  fi
elif [ $SLURM_ARRAY_TASK_ID == 1 ]
then
  seed=2259
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/rational/fulldata/5e-5-r1e-3-2259-frozen
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/original/fulldata/5e-5-2259
  fi
elif [ $SLURM_ARRAY_TASK_ID == 2 ]
then
  seed=5309
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/rational/fulldata/5e-5-r1e-3-5309-frozen
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/original/fulldata/5e-5-5309
  fi
elif [ $SLURM_ARRAY_TASK_ID == 3 ]
then
  seed=202206
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/rational/fulldata/5e-5-r1e-3-202206-frozen
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/original/fulldata/5e-5-202206
  fi
elif [ $SLURM_ARRAY_TASK_ID == 4 ]
then
  seed=20220602
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/rational/fulldata/5e-5-r1e-3-20220602-frozen
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/rte/original/fulldata/5e-5-49
  fi
elif [ $SLURM_ARRAY_TASK_ID == 5 ]
then
  seed=49
  if [[ ${suffix} == "frozen" ]]
  then
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/mnli/rational/fulldata/5e-5-r1e-3-49-frozen
  else
    model_name_or_path=/storage/ukp/work/fang/rational_bert/outputs/cola/original/fulldata/5e-5-49
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
  --output_dir ./outputs/$dataset_name/$activate_func/ \
  --run_name "${dataset_name}-${activate_func}-${seed}-${suffix}" \
  --save_strategy epoch \
  --logging_strategy epoch \
  --save_rational_plots \
  --greater_is_better True \
  --rational_lr ${rational_lr} \
  --report_to none \
  --max_train_samples 1 \
  --overwrite_output_dir \
  --weight_decay 0.1 \
  --seed ${seed} \
  --academicBERT \
  # --zero_shot \
    # --dataset_name ${dataset_name} \
# done