#!/bin/bash
#SBATCH -J gelu-kd
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/kd/imp.err.%x
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/kd/imp.out.%x
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu 3

# change name, logging_steps,metric_for_best_model
activate_func="original"
rational_layers="''"
lr=1e-4
config_name="config_6.json"
layers=$(echo ${config_name}| cut -d'.' -f 1| cut -d'_' -f 2)
optimizer="adam"


echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="./wandb/"
export WANDB_PROJECT="kd"
export WANDB_ENTITY="haishuo"


# torchrun --nproc_per_node=1 \
torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --master_port 6061 \
    run_distillation.py \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --tokenizer_name roberta-base \
    --dataset_name wikipedia \
    --dataset_config_name 20200501.en \
    --config_name ./configs/${config_name} \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate $lr \
    --rational_lr 1e-2 \
    --rational_layers ${rational_layers} \
    --num_train_epochs 3 \
    --eval_accumulation_steps 3200 \
    --report_to wandb \
    --logging_steps 2500 \
    --save_steps 2500 \
    --preprocessing_num_workers 8 \
    --output_dir ./outputs/roberta/kd/$lr/"${layers}layers"/${activate_func}/teacher \
    --approx_func gelu \
    --run_name "kd_${layers}_with_teacher_${activate_func}" \
    --alpha_ce 0.5 \
    --alpha_mlm 0.5 \
    --alpha_cos 0.0 \
    --alpha_clm 0.0 \
    # --overwrite_output_dir \
    # --weight_decay 0.01 \
    