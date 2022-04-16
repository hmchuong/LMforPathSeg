#!/bin/bash
#SBATCH --job-name=cam_0.1
#SBATCH --output=logs/slurm_contrast_unconnected_0.1_%A.out
#SBATCH --error=logs/slurm_contrast_unconnected_0.1_%A.err
#SBATCH --gres=gpu:1
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --nodelist=cmlgrad07
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

module purge
module load cuda/11.1.1
source /fs/classhomes/spring2022/cmsc828l/c828l028/.bashrc
conda activate semseg
now=$(date +"%Y%m%d_%H%M%S")
ROOT=../..
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$ROOT:$PYTHONPATH


python ../../train_contrast.py --config=config_contrast_unconnected_0.1.yaml