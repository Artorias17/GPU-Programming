#!/bin/bash

#SBATCH --job-name=Lab_3.2
#SBATCH --account=project_2016196
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/users/aroy/projects/stdout.txt
#SBATCH --error=/users/aroy/projects/stderr.txt

module load cuda/11.5.0

BASE_DIR="/users/aroy/projects/GP25_Labs/Lab_3.2"
EXE_PATH="${BASE_DIR}/sgemm-tiled"
OUTPUT_PATH="${BASE_DIR}/output.txt"

TESTS=("" "100" "537 679 851" "4859 6075 2310")

echo "" > ${OUTPUT_PATH}
for T in "${TESTS[@]}"; do
    srun ${EXE_PATH} ${T} >> ${OUTPUT_PATH}
    echo "" >> ${OUTPUT_PATH}
done