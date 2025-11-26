#!/bin/bash

#SBATCH --job-name=Lab_2
#SBATCH --account=project_2016196
#SBATCH --partition=gputest
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500M
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/users/aroy/projects/stdout.txt
#SBATCH --error=/users/aroy/projects/stderr.txt

module load cuda/11.5.0

BASE_DIR="/users/aroy/projects/GP25_Labs/Lab_2"
EXE_PATH="${BASE_DIR}/vecadd"
OUTPUT_PATH="${BASE_DIR}/output.txt"
N=1000000

srun ${EXE_PATH} > ${OUTPUT_PATH}
srun ${EXE_PATH} ${N} >> ${OUTPUT_PATH}