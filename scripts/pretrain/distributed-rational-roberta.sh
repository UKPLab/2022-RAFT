#!/bin/bash
#SBATCH -J rational-12layers-bc-nocuda-6e-4
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/pretrain/imp.err.%x
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/pretrain/imp.out.%x
#SBATCH -n 1
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu 3


activate_func="rational"
rational_layers="0-11"
lr=6e-4
config_name="config_12.json"
layers=$(echo ${config_name}| cut -d'.' -f 1| cut -d'_' -f 2)
optimizer="adam"

echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="Pretraining_rational"

module load cuda/11.1
# change name, config file, batch size, graident steps,  logging_steps, config:wikitext-2-raw-v1, 
torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --master_port 6060 \
    run_mlm.py \
    --model_type roberta \
    --config_name ./outputs/roberta/${config_name} \
    --tokenizer_name roberta-base \
    --dataset_name bookcorpus \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --num_train_epochs 50 \
    --report_to wandb \
    --logging_steps 2500 \
    --save_steps 2500 \
    --max_seq_length 128 \
    --learning_rate $lr \
    --preprocessing_num_workers 8 \
    --output_dir ./outputs/roberta/$activate_func/${WANDB_PROJECT}/$lr/"${layers}layers"/"${optimizer}"/${rational_layers}-nocuda-6e-4 \
    --rational_layers $rational_layers \
    --approx_func gelu \
    --save_rational_plots \
    --run_name "${activate_func}-${rational_layers}-${lr}" \
    --eval_accumulation_steps 7500 \
    --seed 202204
    
    # --logging_strategy epoch \
    # --save_strategy epoch \
    # --add_ln True
    # --config_overrides ../output_models/roberta/roberta_config.json \
        # --dataset_config_name 20200501.en \
            # --tb_dir ./runs/graident_check \
        