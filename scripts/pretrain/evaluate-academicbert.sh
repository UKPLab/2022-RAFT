#!/bin/bash
#SBATCH -J evaluate-original
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/pretrain/acabert/%x.err
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/pretrain/acabert/%x.out
#SBATCH -n 1
#SBATCH --mem=128G 
#SBATCH --gres=gpu:1
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
export WANDB_PROJECT="academicBERT"
export WANDB_ENTITY="haishuo"
# export MAX_JOBS=1
# nvidia-smi
# deepspeed 
torchrun --nproc_per_node=1 \
    --nnodes=1 \
    --master_port 6070 \
    run_mlm.py \
    --model_type bert \
    --dataset_name openwebtext \
    --model_name_or_path /ukp-storage-1/fang/rational_bert/outputs/acabert/original/7e-4/12layers/adamw/largedata/checkpoint-16000 \
    --config_name ./configs/${config_name} \
    --tokenizer_name bert-large-uncased \
    --per_device_eval_batch_size 256 \
    --do_eval \
    --optimizer ${optimizer} \
    --report_to none \
    --max_seq_length 128 \
    --curve linear linear linear \
    --lr_schedule step step step \
    --warmup_proportion_list 0.06 0.06 0.06 \
    --preprocessing_num_workers 25 \
    --output_dir ./outputs/acabert/$activate_func/$lr/"${layers}layers"/$optimizer/builtin-mp \
    --rational_layers $rational_layers \
    --approx_func gelu \
    --run_name "academicbert-${activate_func}-${layers}layers-${optimizer}-builtin-mp-eval" \
    --weight_decay 0.01 \
    --academicBERT \
    --fp16 True \
    --evaluation_strategy steps \
    --eval_accumulation_steps 100 \
    --max_eval_samples 256 \
    # --eval_steps 12001 \
        # --dataset_config_name 20200501.en \
    # --max_eval_samples 100000 \
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