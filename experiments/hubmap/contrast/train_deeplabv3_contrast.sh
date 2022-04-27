#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
module purge
module load cuda/11.1.1
source /fs/classhomes/spring2022/cmsc828l/c828l028/.bashrc
conda activate semseg
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$ROOT:$PYTHONPATH


python ../../../train_contrast.py --config=config_deeplabv3_contrast.yaml  2>&1 | tee log_contrast_$now.txt
# mkdir -p checkpoints/result
#python ../../eval.py --base_size=2048 --scales 1.0 --config=config.yaml --model_path=checkpoints/best.pth --save_folder=checkpoints/result/ 2>&1 | tee checkpoints/result/eva_$now.log
#python ../../eval.py --base_size=2048 --scales 1.0 --config=config.yaml --model_path=checkpoints/epoch_99.pth --save_folder=checkpoints/result/ 2>&1 | tee checkpoints/result/eva_99_$now.log
#