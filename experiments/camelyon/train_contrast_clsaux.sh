#!/bin/bash
#SBATCH --job-name=cam_dlv3_cls       
#SBATCH --output=logs/slurm_dlv3_contrast_clsaux_%A.out
#SBATCH --error=logs/slurm_dlv3_contrast_clsaux_%A.err
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-12:00:00

now=$(date +"%Y%m%d_%H%M%S")
module purge
module load cuda/11.1.1
source /fs/classhomes/spring2022/cmsc828l/c828l028/.bashrc
conda activate semseg
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$ROOT:$PYTHONPATH


python ../../train_contrast.py --config=config_contrast_clsaux.yaml
# python ../../test.py --config=config_contrast_clsaux.yaml