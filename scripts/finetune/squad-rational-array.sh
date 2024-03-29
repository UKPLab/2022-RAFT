#!/bin/bash
#SBATCH -J rational-squad
#SBATCH -e /storage/ukp/work/fang/rational_bert/logs/finetune/%x.err
#SBATCH -o /storage/ukp/work/fang/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu 3
#SBATCH --array=0-3

#2022,202206,20220602,2259,49,5309
# change name, logging_steps,metric_for_best_model
activate_func="rational"
rational_layers="0-11,pooler"
lr=1e-4
rational_lr=5e-4
dataset_name="squad"
num_ins=full_data
suffix="128seq"

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
fi

echo JOB ID: "${SLURM_JOBID}"
source /storage/ukp/work/fang/miniconda3/bin/activate /storage/ukp/work/fang/miniconda3
module load cuda/11.1
export WANDB_DIR="/storage/ukp/work/fang/rational_bert/wandb_t"
export WANDB_CONFIG_DIR="/storage/ukp/work/fang/rational_bert/wandb_t"
export WANDB_PROJECT="squad"
export WANDB_ENTITY="haishuo"
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/ukp-storage-1/fang/miniconda3/lib/"
# for rnd_seed in 20220422 20220423
# do
# torchrun --nproc_per_node=1 \
#   --nnodes=1 \
#   --master_port 6061 \
python  run_squad.py \
  --model_name_or_path /storage/ukp/work/fang/rational_bert/outputs/acabert/rational/7e-4/12layers/adamw/0.005rlr-constlr-pooler \
  --model_type bert \
  --do_train \
  --do_eval \
  --run_name squad_lr_${lr}_r_${rational_lr}_${seed}_${num_ins}_${suffix} \
  --cache_dir ./data_temp/squad \
  --data_dir ./data_temp/squad \
  --max_seq_length 128 \
  --evaluate_during_training \
  --per_gpu_train_batch_size 32 \
  --curve linear linear linear \
  --per_gpu_eval_batch_size 32 \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 10 \
  --output_dir ./outputs/$dataset_name/$activate_func/$num_ins/${lr}-${rational_lr}-${seed}-${suffix}/ \
  --rational_lr ${rational_lr} \
  --overwrite_output_dir \
  --weight_decay 0.1 \
  --seed ${seed} \
  --academicBERT \
  --logging_steps 0 \
  --save_steps 0 \
  --frozen_rf \
  # --save_rational_plot \
  # \
  # --save_rational_functions 
  # --max_train_samples ${num_ins} \
  # --max_dev_samples ${num_ins} \
# /storage/ukp/work/fang/rational_bert/outputs/squad/rational/full_data/1e-4-5e-4-49- \
    # --run_name "academicbert-${dataset_name}-${activate_func}-${lr}-${seed}-${rational_lr}" \
    # --report_to

# done