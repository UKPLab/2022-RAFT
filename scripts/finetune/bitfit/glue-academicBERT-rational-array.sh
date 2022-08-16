#!/bin/bash
#SBATCH -J rational-qqp-ch-rf
#SBATCH -e /storage/ukp/work/fang/rational_bert/logs/finetune/%x.err
#SBATCH -o /storage/ukp/work/fang/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#2022,202206,20220602,2259,49,5309
# change name, logging_steps,metric_for_best_model
activate_func="rational"
rational_layers="0-11,pooler"
lr=5e-3
rational_lr=5e-3
task_name="qqp"
num_ins=fulldata
suffix="ch_bit"


if [ $SLURM_ARRAY_TASK_ID == 0 ]
then
  seed=2022
elif [ $SLURM_ARRAY_TASK_ID == 1 ]
then
  seed=202206
elif [ $SLURM_ARRAY_TASK_ID == 2 ]
then
  seed=20220602
elif [ $SLURM_ARRAY_TASK_ID == 3 ]
then
  seed=2259
elif [ $SLURM_ARRAY_TASK_ID == 4 ]
then
  seed=5309

elif [ $SLURM_ARRAY_TASK_ID == 5 ]
then
  seed=49
fi



echo JOB ID: "${SLURM_JOBID}"
source /storage/ukp/work/fang/miniconda3/bin/activate /storage/ukp/work/fang/miniconda3
module load cuda/11.1
export WANDB_CONFIG_DIR="/storage/ukp/work/fang/rational_bert/wandb/"
export WANDB_PROJECT="glue-deepnet-ht-new"
export WANDB_ENTITY="haishuo"

# for rnd_seed in 20220422 20220423
# do





# torchrun --nproc_per_node=1 \
#   --nnodes=1 \
#   --master_port 6062 \
python run_glue.py \
  --model_name_or_path /storage/ukp/work/fang/rational_bert/outputs/acabert/rational/7e-4/12layers/adamw/0.005rlr-constlr-pooler \
  --model_type bert \
  --task_name $task_name \
  --cache_dir ./data_temp \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 128 \
  --curve linear linear linear \
  --lr_schedule step step step \
  --num_warmup_steps 0 0 0 \
  --evaluation_strategy epoch \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 6 \
  --output_dir ./outputs/$task_name/$activate_func/${num_ins}/${lr}-r${rational_lr}-${seed}-${suffix} \
  --run_name "${task_name}-${activate_func}-${lr}-r${rational_lr}-${seed}-${num_ins}-${suffix}" \
  --save_strategy epoch \
  --logging_strategy epoch \
  --save_rational_plots \
  --rational_lr ${rational_lr} \
  --report_to wandb \
  --weight_decay 0.1 \
  --seed ${seed} \
  --academicBERT \
  --save_total_limit 2 \
  --load_best_model_at_end \
  --metric_for_best_model eval_accuracy \
  --greater_is_better True \
  --patience 10 \
  --sub_bias \
      # --overwrite_output_dir \
  # --max_train_samples ${num_ins} \
  # --max_eval_samples ${num_ins} \

  
  


# done