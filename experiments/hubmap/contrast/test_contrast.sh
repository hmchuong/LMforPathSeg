#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
ROOT=../..
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$ROOT:$PYTHONPATH


python ../../../test.py --config=config_contrast.yaml  2>&1 | tee log_test_contrast_$now.txt