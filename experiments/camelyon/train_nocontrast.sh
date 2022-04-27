#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
ROOT=../..
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$ROOT:$PYTHONPATH


python ../../train_contrast.py --config=config_nocontrast.yaml  2>&1 | tee log_nocontrast_$now.txt
# mkdir -p checkpoints/result
#python ../../eval.py --base_size=2048 --scales 1.0 --config=config.yaml --model_path=checkpoints/best.pth --save_folder=checkpoints/result/ 2>&1 | tee checkpoints/result/eva_$now.log
#python ../../eval.py --base_size=2048 --scales 1.0 --config=config.yaml --model_path=checkpoints/epoch_99.pth --save_folder=checkpoints/result/ 2>&1 | tee checkpoints/result/eva_99_$now.log
#