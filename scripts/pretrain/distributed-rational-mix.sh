#!/bin/bash
#SBATCH -J rational-12layers-bc-mix-6-11-trainable
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/pretrain/imp.err.%x
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/pretrain/imp.out.%x
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --gres=gpu:2

activate_func="mix"
rational_layers="0,2,4,6,8,10"
lr=1e-4
config_name="config_12.json"
layers=$(echo ${config_name}| cut -d'.' -f 1| cut -d'_' -f 2)
optimizer="adam"
origin_stat = "trainable"

echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="Pretraining"

# module load cuda/11.1
# change name, config file, batch size, graident steps,  logging_steps, config:wikitext-2-raw-v1, 
torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --master_port 6062 \
    run_mlm.py \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --tokenizer_name roberta-base \
    --dataset_name bookcorpus \
    --config_name ./configs/${config_name} \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --num_train_epochs 3 \
    --report_to none \
    --logging_steps 2500 \
    --save_steps 2500 \
    --max_seq_length 128 \
    --learning_rate $lr \
    --preprocessing_num_workers 10 \
    --output_dir ./outputs/roberta/$activate_func/$lr/"${layers}layers"/"${optimizer}"/${rational_layers}-${lr}-${origin_stat} \
    --rational_layers $rational_layers \
    --approx_func gelu \
    --save_rational_plots \
    --run_name "${activate_func}-${config_name}-${optimizer}-${rational_layers}-${lr}-${origin_stat}" \
    --eval_accumulation_steps 2500 \
    --do_mix \
    --pretrained_layers 0-5 \
    # --freeze_pl \
    # --logging_strategy epoch \
    # --save_strategy epoch \
    # --add_ln True
    # --config_overrides ../output_models/roberta/roberta_config.json \
        # --dataset_config_name 20200501.en \
        
        