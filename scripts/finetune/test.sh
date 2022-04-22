#!/bin/bash
#SBATCH -J rational-test
#SBATCH -e /ukp-storage-1/fang/pretrain_bert/logs/imp.err.%x
#SBATCH -o /ukp-storage-1/fang/pretrain_bert/logs/imp.out.%x
#SBATCH --mem=50G 
#SBATCH --gres=gpu:1

python test.py