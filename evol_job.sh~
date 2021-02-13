#!/bin/bash -l
#SBATCH --job-name=BalancedVersion
#SBATCH --partition=gpu
#SBATCH --output=job-%j.out
#SBATCH --mem 32000M
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:4
#SBATCH --mail-user=cwhite31@qub.ac.uk
#SBATCH --mail-type=END

module add nvidia-cuda
module add apps/python3

nvidia-smi

export PYTHONPATH=$PYTHONPATH:/users/40175159/gridware/share/python/3.6.4/lib/python3.6/site-packages

python3 main.py --device cuda --population_size 4

