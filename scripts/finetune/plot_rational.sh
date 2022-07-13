#!/bin/bash
#SBATCH -J plot-mnli
#SBATCH -e /ukp-storage-1/fang/rational_bert/logs/finetune/%x.err
#SBATCH -o /ukp-storage-1/fang/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1

#2022,202206,20220602,2259,49,5309
# change name, logging_steps,metric_for_best_model
activate_func="rational"
rational_layers="0-11,pooler"
lr=5e-5
rational_lr=1e-3
task_name="stsb"
num_ins=100
seed=49

if [ $SLURM_ARRAY_TASK_ID == 0 ]
then
  task_name="stsb"
  model_path="/ukp-storage-1/fang/rational_bert/outputs/stsb/rational/100/5e-5-r5e-3-49/checkpoint-44"
  
elif [ $SLURM_ARRAY_TASK_ID == 1 ]
then
  task_name="qnli"
  model_path="/ukp-storage-1/fang/rational_bert/outputs/qnli/rational/100/5e-5-r5e-3-49/checkpoint-36"
elif [ $SLURM_ARRAY_TASK_ID == 2 ]
then
  task_name="mnli"
  model_path="/ukp-storage-1/fang/rational_bert/outputs/mnli/rational/100/5e-5-r5e-4-49/checkpoint-80"

elif [ $SLURM_ARRAY_TASK_ID == 3 ]
then
  task_name="sst2"
  model_path="/ukp-storage-1/fang/rational_bert/outputs/sst2/rational/100/5e-5-r5e-3-49/checkpoint-28"
elif [ $SLURM_ARRAY_TASK_ID == 4 ]
then
  task_name="cola"
  model_path="/ukp-storage-1/fang/rational_bert/outputs/cola/rational/100/5e-5-r5e-4-49/checkpoint-64"
fi



echo JOB ID: "${SLURM_JOBID}"
source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
module load cuda/11.1

# for rnd_seed in 20220422 20220423
# do

python3 run_glue.py \
  --model_name_or_path ${model_path} \
  --model_type bert \
  --task_name $task_name \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --curve linear linear linear \
  --lr_schedule step step step \
  --num_warmup_steps 0 0 0 \
  --evaluation_strategy epoch \
  --learning_rate $lr \
  --rational_layers $rational_layers \
  --num_train_epochs 20 \
  --output_dir ./outputs/$task_name/$activate_func/${num_ins}/${lr}-r${rational_lr}-${seed} \
  --run_name "${task_name}-${activate_func}-${lr}-r${rational_lr}-${seed}-${num_ins}" \
  --save_strategy epoch \
  --logging_strategy epoch \
  --save_rational_plots \
  --rational_lr ${rational_lr} \
  --report_to none \
  --weight_decay 0.1 \
  --seed ${seed} \
  --academicBERT \
  --max_train_samples ${num_ins} \
  --max_eval_samples ${num_ins} \
  --patience 10 \
  --overwrite_output_dir \