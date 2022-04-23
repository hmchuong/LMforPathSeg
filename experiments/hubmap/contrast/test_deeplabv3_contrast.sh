#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
module load cuda/11.1.1
source /fs/classhomes/spring2022/cmsc828l/c828l028/.bashrc
conda activate semseg
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$ROOT:$PYTHONPATH


python ../../../test.py --config=config_deeplabv3_contrast.yaml  2>&1 | tee log_test_dlv3_contrast_zeros_$now.txt