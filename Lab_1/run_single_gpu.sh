#!/bin/bash

#SBATCH --job-name=single_gpu
#SBATCH --account=project_2016196
#SBATCH --partition=gpusmall
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --output=/users/aroy/projects/stdout.txt
#SBATCH --error=/users/aroy/projects/stderr.txt


module swap gcc/11.2.0 gcc/10.4.0
module load cuda/12.6.1
srun /users/aroy/projects/GP25_Labs/Lab_1/device-query > /users/aroy/projects/multi_gpu.txt