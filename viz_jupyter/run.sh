#!/bin/bash
#SBATCH -A m4874
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --cpus-per-task=128
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH -o %x-%j.out

eval "$(conda shell.bash hook)"
conda activate /pscratch/sd/y/yy6080/ocean_sim

# Movies — separate calls to avoid one blocking the other
# python OM4_reso_all.py --freq yearly  --var temperature
# python OM4_reso_all.py --freq yearly  --var ohc
# python OM4_reso_all.py --freq monthly --var temperature
# python OM4_reso_all.py --freq monthly --var ohc
python OM4_reso_all.py --freq all --var temperature
# python OM4_reso_all.py --freq all --var ohc

