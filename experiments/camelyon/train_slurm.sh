#!/bin/bash
<<<<<<< HEAD
#SBATCH --job-name=con0112
#SBATCH --output=slurm_%A_contrast0112.out
=======
<<<<<<< HEAD
#SBATCH --job-name=HM
#SBATCH --output=slurm_%A.out
>>>>>>> a5b27b77b9faad6ac54dd964a7ae1bbdabda6bd4
#SBATCH --error=slurm_%A.err
=======
#SBATCH --job-name=cam_0.1
#SBATCH --output=logs/slurm_contrast_unconnected_0.1_%A.out
#SBATCH --error=logs/slurm_contrast_unconnected_0.1_%A.err
>>>>>>> 68fd43650011bc674dd8f8d3bca6d8a4aa19e8d6
#SBATCH --gres=gpu:1
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --nodelist=cmlgrad07
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
<<<<<<< HEAD
#SBATCH --time=1-12:00:00
=======
<<<<<<< HEAD
#SBATCH --time=16:00:00
=======
#SBATCH --time=24:00:00
>>>>>>> 68fd43650011bc674dd8f8d3bca6d8a4aa19e8d6
>>>>>>> a5b27b77b9faad6ac54dd964a7ae1bbdabda6bd4

module purge
module load cuda/11.1.1
source /fs/classhomes/spring2022/cmsc828l/c828l028/.bashrc
conda activate semseg
now=$(date +"%Y%m%d_%H%M%S")
ROOT=../..
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$ROOT:$PYTHONPATH


<<<<<<< HEAD
python ../../train_contrast.py --config=/fs/classhomes/spring2022/cmsc828l/c828l050/RegionContrast-Med/experiments/camelyon/config_contrast.yaml
=======
<<<<<<< HEAD
python ../../train_contrast.py --config=/fs/classhomes/spring2022/cmsc828l/c828l050/RegionContrast-Med/experiments/camelyon/config_contrast_HM.yaml
=======
python ../../train_contrast.py --config=config_contrast_unconnected_0.1.yaml
>>>>>>> 68fd43650011bc674dd8f8d3bca6d8a4aa19e8d6
>>>>>>> a5b27b77b9faad6ac54dd964a7ae1bbdabda6bd4
