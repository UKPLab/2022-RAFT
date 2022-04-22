#!/bin/bash
#SBATCH -J rational-recadam-rte-2e-5withoutln
#SBATCH -e /ukp-storage-1/fang/pretrain_bert/logs/finetune/imp.err.%x
#SBATCH -o /ukp-storage-1/fang/pretrain_bert/logs/finetune/imp.out.%x
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1

# change name, logging_steps,metric_for_best_model
activate_func="rational"
rational_layers="0-11"
lr=2e-5
task_name="rte"


echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="glue"


# torchrun --nproc_per_node=1 \
python run_glue.py \
  --model_name_or_path  roberta-base \
  --model_type roberta \
  --task_name $task_name \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --evaluation_strategy epoch \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 20 \
  --output_dir ./outputs/$task_name/$activate_func/recadam-2e-5-withoutln \
  --run_name "${task_name}-${activate_func}-recadam" \
  --save_strategy epoch \
  --logging_steps 78 \
  --save_rational_plots \
  --load_best_model_at_end \
  --metric_for_best_model eval_accuracy \
  --greater_is_better True \
  --rational_lr 2e-5 \
  --max_train_samples 1 \
  --report_to wandb \
  --optimizer recadam \
  --overwrite_output_dir \