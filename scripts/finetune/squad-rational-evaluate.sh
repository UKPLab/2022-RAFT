#!/bin/bash
#SBATCH -J rational-squad-eval
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/finetune/%x.err
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=50G 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu 3

# change name, logging_steps,metric_for_best_model
activate_func="rational"
rational_layers="0-11"
lr=5e-5
rational_lr=2e-4
dataset_name="squad"
seed=49


echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="glue-academicBERT"
export WANDB_ENTITY="haishuo"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/ukp-storage-1/fang/miniconda3/lib/"
# for rnd_seed in 20220422 20220423
# do
# torchrun --nproc_per_node=2 \
#   --nnodes=1 \
#   --master_port 6060 \
python -m torch.distributed.launch --nproc_per_node=1 \
  run_squad.py \
  --model_name_or_path /ukp-storage-1/fang/rational_bert/outputs/squad/rational/5e-5/checkpoint-10000 \
  --model_type bert \
  --do_eval \
  --cache_dir ./data/squad \
  --data_dir ./data/squad \
  --max_seq_length 128 \
  --evaluate_during_training \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 5 \
  --output_dir ./outputs/$dataset_name/$activate_func/${lr}/ \
  --save_steps 2000 \
  --logging_steps 500 \
  --rational_lr ${rational_lr} \
  --overwrite_output_dir \
  --weight_decay 0.01 \
  --seed ${seed} \
  --academicBERT \
    # --run_name "academicbert-${dataset_name}-${activate_func}-${lr}-${seed}-${rational_lr}" \
    # --report_to

# done