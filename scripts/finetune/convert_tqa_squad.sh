#!/bin/bash
#SBATCH -J convert-tqa-squad
#SBATCH -e /storage/ukp/work/fang/rational_bert/logs/finetune/%x.err
#SBATCH -o /storage/ukp/work/fang/rational_bert/logs/finetune/%x.out
#SBATCH -n 1
#SBATCH --mem=120G 

python ./triviaqa/convert_to_squad_format.py \
--triviaqa_file ./data_temp/triviaqa/qa/verified-wikipedia-dev.json \
--squad_file ./data_temp/triviaqa/qa/verified-wikipedia-dev-s.json \
--web_dir ./data_temp/triviaqa/evidence/web \
--wikipedia_dir ./data_temp/triviaqa/evidence/wikipedia \
