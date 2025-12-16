#!/bin/bash
#SBATCH --job-name=yy_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=eval_%j.out

export VIRTUAL_ENV=/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.venv
source $VIRTUAL_ENV/bin/activate

NAME1="2025-12-03-samudra_om4_onedeg"
NAME2="2025-12-04-samudra_om4_onedeg_blur_eval"

NAME11="samudra_om4_onedeg"
NAME22="samudra_om4_onedeg_blur"

DATA_PATH=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg

uv run --preview-features extra-build-dependencies -m ocean_emulators.viz configs/samudra_om4/viz.yaml \
    --data_root $DATA_PATH \
    --basins_location "basin_masks_original.zarr" \
    --name "${NAME22}_viz" \
    --runs="[{\"name\": \"$NAME11\", \"location\": \"/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.LOCAL/${NAME1}/predictions.zarr\"}, {\"name\": \"$NAME22\", \"location\": \"/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.LOCAL/${NAME2}/predictions.zarr\"}]" \
    
# sbatch -A m4874 -C cpu -q regular scripts/run_eval_onedeg_blur_cmp.sbatch