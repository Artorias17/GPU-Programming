#!/bin/bash

#SBATCH --job-name=multi_gpu
#SBATCH --account=project_2016196
#SBATCH --partition=gputest
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:a100:4
#SBATCH --output=/users/aroy/projects/stdout.txt
#SBATCH --error=/users/aroy/projects/stderr.txt


module swap gcc/11.2.0 gcc/10.4.0
module load cuda/12.6.1
srun /users/aroy/projects/GP25_Labs/Lab_1/device-query > /users/aroy/projects/multi_gpu.txt