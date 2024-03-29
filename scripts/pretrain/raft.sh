#!/bin/bash
#SBATCH -J raft-gelu-500k
#SBATCH -e /storage/ukp/work/fang/rational_bert/logs/pretrain/acabert/%x.err
#SBATCH -o /storage/ukp/work/fang/rational_bert/logs/pretrain/acabert/%x.out
#SBATCH -n 1
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu 3


# export CXX=g++
activate_func="rational"
rational_layers="0-11,pooler"
lr=6e-4
rlr=1e-4
config_name="config_12_academic_postln.json"
layers=$(echo ${config_name}| cut -d'.' -f 1| cut -d'_' -f 2)
optimizer="adamw"
run_name="raft-${lr}-r${rlr}-8192-gelu-500k"

# change name, config file, batch size, graident steps,  logging_steps, 

echo JOB ID: "${SLURM_JOBID}"
source /storage/ukp/work/fang/miniconda3/bin/activate /storage/ukp/work/fang/miniconda3
module load cuda/11.1
export WANDB_PROJECT="Overfitting-rational"
export WANDB_CONFIG_DIR="/storage/ukp/work/fang/rational_bert/wandb/"



torchrun --nproc_per_node=2 \
    --nnodes=1 \
    --master_port 6067 \
    run_mlm.py \
    --model_type bert \
    --dataset_name wikipedia \
    --dataset_config_name 20200501.en \
    --config_name ./configs/${config_name} \
    --tokenizer_name bert-large-uncased \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 256 \
    --do_train \
    --do_eval \
    --max_steps 500000 \
    --curve constant linear linear \
    --lr_schedule step step step \
    --warmup_proportion_list 0.048 0.048 0.048 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --half_precision_backend auto \
    --num_train_epochs 1000 \
    --optimizer ${optimizer} \
    --logging_steps 250 \
    --save_steps 2000 \
    --save_total_limit 15 \
    --max_seq_length 128 \
    --learning_rate $lr \
    --rational_lr ${rlr} \
    --preprocessing_num_workers 15 \
    --rational_layers $rational_layers \
    --approx_func gelu \
    --run_name $run_name \
    --weight_decay 0.01 \
    --academicBERT \
    --fp16 True \
    --evaluation_strategy steps \
    --eval_accumulation_steps 30 \
    --eval_steps 250 \
    --report_to wandb \
    --max_grad_norm 0 \
    --output_dir ./outputs/acabert/$activate_func/$lr/"${layers}layers"/$optimizer/${run_name}/ \
    --max_eval_samples 50000 \
        # --save_rational_plots \
