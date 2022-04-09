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

#module purge
#module load cuda/11.1.1
#source /fs/classhomes/spring2022/cmsc828l/c828l028/.bashrc
#conda activate semseg
now=$(date +"%Y%m%d_%H%M%S")
ROOT=../..
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$ROOT:$PYTHONPATH


python ../../train_contrast.py --config=/fs/classhomes/spring2022/cmsc828l/c828l050/RegionContrast-Med/experiments/camelyon/config_nocontrast.yaml  2>&1 | tee log_$now.txt