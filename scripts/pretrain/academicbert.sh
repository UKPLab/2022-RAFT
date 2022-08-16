#!/bin/bash
#SBATCH -J original
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/pretrain/acabert/%x.err
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/pretrain/acabert/%x.out
#SBATCH -n 1
#SBATCH --mem=128G 
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu 3


export CXX=g++
export CUDA_LAUNCH_BLOCKING=1
activate_func="original"
rational_layers="''"
lr=7e-4
config_name="config_12_academic_postln.json"
layers=$(echo ${config_name}| cut -d'.' -f 1| cut -d'_' -f 2)
optimizer="adamw"

# change name, config file, batch size, graident steps,  logging_steps, 

echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="/ukp-storage-1/fang/rational_bert/wandb/"
export WANDB_PROJECT="Overfitting-rational"
export WANDB_ENTITY="haishuo"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
# export MAX_JOBS=1
# nvidia-smi
# deepspeed 
torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --master_port 6060 \
    run_mlm.py \
    --model_type bert \
    --dataset_name wikipedia \
    --config_name ./configs/${config_name} \
    --dataset_config_name 20200501.en \
    --tokenizer_name bert-large-uncased \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 256 \
    --do_train \
    --do_eval \
    --lr_schedule step step step \
    --max_steps 23000 \
    --curve linear linear linear \
    --warmup_proportion 0.01 0.01 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --half_precision_backend auto \
    --num_train_epochs 100 \
    --optimizer ${optimizer} \
    --report_to wandb \
    --save_total_limit 20 \
    --logging_steps 250 \
    --save_steps 1000 \
    --max_seq_length 128 \
    --learning_rate $lr \
    --preprocessing_num_workers 15 \
    --output_dir ./outputs/acabert/$activate_func/$lr/"${layers}layers"/$optimizer/deepnet-0.01warm \
    --rational_layers $rational_layers \
    --approx_func gelu \
    --run_name "academicbert-${activate_func}-${layers}layers-${lr}-${optimizer}-deepnet-0.01warm" \
    --weight_decay 0.01 \
    --academicBERT \
    --fp16 True \
    --evaluation_strategy steps \
    --eval_accumulation_steps 30 \
    --eval_steps 250 \
    --max_grad_norm 0 \
    --max_eval_samples 50000 \
        # --dataset_config_name 20200501.en \
    #
    #     # 
        # 
    #    --dataset_config_name 20200501.en \
    # --deepspeed ./configs/deepspeed.json \
    # --max_train_samples 2000
    #   --overwrite_output_dir \
   
    # --stochastic_mode \
    # --deepspeed_transformer_kernel \
        
    # --tb_dir ./runs/graident_check-original \
    # --dataset_config_name 20200501.en \
    # --add_ln True
    # --config_overrides ../output_models/roberta/roberta_config.json \