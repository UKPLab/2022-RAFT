#!/bin/bash
#SBATCH -J rational-squad-deepnet
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/finetune/%x.err
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=50G 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu 3

#2022,202206,20220602,2259,49,5309
# change name, logging_steps,metric_for_best_model
activate_func="rational"
rational_layers="0-11"
lr=2e-4
rational_lr=0.01
dataset_name="squad"
seed=5309


echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="squad"
export WANDB_ENTITY="haishuo"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/ukp-storage-1/fang/miniconda3/lib/"
# for rnd_seed in 20220422 20220423
# do
# torchrun --nproc_per_node=4 \
#   --nnodes=1 \
#   --master_port 6060 \
python run_squad.py \
  --model_name_or_path /ukp-storage-1/fang/rational_bert/outputs/acabert/rational/7e-4/12layers/adamw/academicrational-0.005rlr-deepnet \
  --model_type bert \
  --do_train \
  --do_eval \
  --run_name squad_lr_${lr}_rational_${rational_lr}_${seed} \
  --cache_dir ./data/squad \
  --data_dir ./data/squad \
  --max_seq_length 384 \
  --evaluate_during_training \
  --per_gpu_train_batch_size 32 \
  --curve linear linear linear \
  --per_gpu_eval_batch_size 32 \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 8 \
  --output_dir ./outputs/$dataset_name/$activate_func/${lr}-${rational_lr}-${seed}/ \
  --rational_lr ${rational_lr} \
  --overwrite_output_dir \
  --weight_decay 0.1 \
  --seed ${seed} \
  --academicBERT \
  --max_train_samples 1000 \
  --max_dev_samples 1000 \
  --logging_steps 33 \
  --save_steps 33 \
  --save_rational_plot \

    # --run_name "academicbert-${dataset_name}-${activate_func}-${lr}-${seed}-${rational_lr}" \
    # --report_to

# done