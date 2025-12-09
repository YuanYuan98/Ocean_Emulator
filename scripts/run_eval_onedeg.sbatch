#!/bin/bash
#SBATCH --job-name=yy_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=eval_%j.out

NAME="2025-12-03-samudra_om4_onedeg"

DATA_PATH=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg

CKPT=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/$NAME/ema_ckpt.pt

CONFIG_PATH=configs/samudra_om4/eval.yaml

uv run --preview-features extra-build-dependencies -m ocean_emulators.eval \
    $CONFIG_PATH \
    --ckpt_path $CKPT \
    --experiment.data_root $DATA_PATH \
    --experiment.name $NAME \
    --save_zarr true

uv run --preview-features extra-build-dependencies -m ocean_emulators.viz configs/samudra_om4/viz.yaml \
    --data_root $DATA_PATH \
    --name "${NAME}_viz" \
    --runs="[{\"name\": \"$NAME\", \"location\": \"/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.LOCAL/${NAME}/predictions.zarr\"}]"

# sbatch -A m4874 -C gpu -q debug scripts/run_eval.sbatch
# 2025-12-02-samudra_om4_halfdeg  2025-12-03-samudra_om4_onedeg