#!/bin/bash
#SBATCH -A m4874
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH -o %x-%j.out

eval "$(conda shell.bash hook)"
conda activate /pscratch/sd/y/yy6080/ocean_sim

python OM4_reso_all.py