#!/bin/bash
#SBATCH --job-name=yy_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=eval_%j.out

export VIRTUAL_ENV=/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.venv
source $VIRTUAL_ENV/bin/activate

NAME="2025-12-02-samudra_om4_halfdeg"

DATA_PATH=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/halfdeg

CKPT=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/$NAME/ema_ckpt.pt

CONFIG_PATH=configs/samudra_om4/eval.yaml

# uv run --preview-features extra-build-dependencies -m ocean_emulators.eval \
#     $CONFIG_PATH \
#     --ckpt_path $CKPT \
#     --experiment.data_root $DATA_PATH \
#     --experiment.name $NAME \
#     --save_zarr true

uv run --preview-features extra-build-dependencies -m ocean_emulators.viz configs/samudra_om4/viz.yaml \
    --data_root $DATA_PATH \
    --basins_location "basin_masks_regridded.zarr" \
    --name "${NAME}_viz" \
    --runs="[{\"name\": \"$NAME\", \"location\": \"/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.LOCAL/${NAME}/predictions.zarr\"}]" \
    
# sbatch -A m4874 -C gpu -q regular scripts/run_eval_halfdeg.sbatch
# 2025-12-02-samudra_om4_halfdeg  2025-12-03-samudra_om4_onedeg