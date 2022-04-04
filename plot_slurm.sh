#!/bin/bash
#SBATCH --job-name=train_camelyon16
#SBATCH --output=slurm_%A.out
#SBATCH --error=slurm_%A.err
#SBATCH --gres=gpu:1
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$ROOT:$PYTHONPATH


python plot_cancerous.py