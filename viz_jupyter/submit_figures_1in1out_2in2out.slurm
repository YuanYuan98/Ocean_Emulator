#!/bin/bash
#SBATCH -J figures_1in1out_2in2out
#SBATCH -A m4874
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -o slurm_logs/figures_1in1out_2in2out_%j.out
#SBATCH -e slurm_logs/figures_1in1out_2in2out_%j.err

set -euo pipefail

cd /global/u2/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/viz_jupyter

module load cpu || true

source /pscratch/sd/y/yy6080/miniconda3/bin/activate /pscratch/sd/y/yy6080/ocean_figures_xesmf

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python Figures_1in1out_2in2out.py
