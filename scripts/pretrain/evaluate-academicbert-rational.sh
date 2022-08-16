#!/bin/bash
#SBATCH -J evaluate-rational-plot
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/pretrain/evaluate/%x.err
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/pretrain/evaluate/%x.out
#SBATCH -n 1
#SBATCH --mem=128G 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu 3


export CXX=g++
export CUDA_LAUNCH_BLOCKING=1
activate_func="rational"
rational_layers="0-11,pooler"
lr=7e-4
rational_lr=0.005
config_name="config_12_academic_postln.json"
layers=$(echo ${config_name}| cut -d'.' -f 1| cut -d'_' -f 2)
optimizer="adamw"

# change name, config file, batch size, graident steps,  logging_steps, 

echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="/ukp-storage-1/fang/rational_bert/wandb/"
export WANDB_PROJECT="academicBERT_evaluate"
export WANDB_ENTITY="haishuo"
# export MAX_JOBS=1
# nvidia-smi
# deepspeed 
torchrun --nproc_per_node=1 \
    --nnodes=1 \
    --master_port 6060 \
    run_mlm.py \
    --model_type bert \
    --dataset_name wikipedia \
    --dataset_config_name 20200501.en \
    --model_name_or_path /ukp-storage-1/fang/rational_bert/outputs/acabert/rational/7e-4/12layers/adamw/0.005rlr-constlr-pooler \
    --config_name ./configs/${config_name} \
    --tokenizer_name bert-large-uncased \
    --per_device_eval_batch_size 256 \
    --do_eval \
    --optimizer ${optimizer} \
    --report_to wandb \
    --max_seq_length 128 \
    --learning_rate $lr \
    --curve linear linear linear \
    --lr_schedule step step step \
    --warmup_proportion_list 0.06 0.06 0.06 \
    --preprocessing_num_workers 15 \
    --output_dir ./outputs/acabert/$activate_func/$lr/"${layers}layers"/$optimizer/builtin-mp \
    --rational_layers $rational_layers \
    --approx_func gelu \
    --run_name "deepnet-${lr}-r${rational_lr}-constlr-pooler" \
    --weight_decay 0.01 \
    --academicBERT \
    --fp16 True \
    --evaluation_strategy steps \
    --eval_accumulation_steps 100 \
    --save_rational_plots \
    --max_eval_samples 1 \
    # --eval_steps 12001 \
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