#!/bin/bash
#SBATCH -J rational-0.0001-eval
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/imp.err.%x
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/imp.out.%x
#SBATCH -n 1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1

activate_func="rational"
rational_layers="0-11"
lr=1e-4
echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
# module load cuda/11.1

python run_mlm.py \
    --model_name_or_path /ukp-storage-1/fang/pretrain_bert/outputs/roberta/rational/0.01/checkpoint-29509 \
    --dataset_name wikitext \
    --config_name ./configs/config_large.json \
    --tokenizer_name roberta-base \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_eval \
    --num_train_epochs 50 \
    --logging_steps 1283 \
    --tb_dir ./runs/$activate_func \
    --save_steps  1283 \
    --save_rational_plots \
    --max_seq_length 128 \
    --learning_rate $lr \
    --preprocessing_num_workers 4 \
    --output_dir ./outputs/roberta/$activate_func/$lr \
    --rational_layers $rational_layers \
    --approx_func gelu \
    --eval_accumulation_steps 10 \
    --report_to tensorboard 
    # --overwrite_output_dir True \
    # --add_ln True
    # --config_overrides ../output_models/roberta/roberta_config.json \