#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gpus=A40:1
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/%x_job_%j.out
#SBATCH --job-name=oa_reactdiff
#SBATCH --partition=fastlane

module load Miniconda3
source ${EBROOTMINICONDA3}/bin/activate
conda activate reactot

echo "Running on host $(hostname)"
echo "Time is $(date)"
echo "Current directory is $(pwd)"

timesteps=250
resamplings=10
jump_length=10
partition="valid_addprop"
single_frag_only=0
model="leftnet_new_encoder"
power="2.0"
repeats=1

echo "Starting evaluation..."

python -u evaluate_ts_w_rp.py \
    --timesteps $timesteps \
    --resamplings $resamplings \
    --jump_length $jump_length \
    --partition $partition \
    --single_frag_only $single_frag_only \
    --model $model \
    --power $power \
    --repeats $repeats