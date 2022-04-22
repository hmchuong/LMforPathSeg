#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
ROOT=../..
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$ROOT:$PYTHONPATH


python ../../../train_contrast.py --config=config_unetseresnext101_contrast.yaml  2>&1 | tee log_train_unetseresnext101_contrast_$now.txt