#!/bin/bash
#SBATCH --job-name=yy_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=eval_%j.out

# NAME_long="2025-12-03-samudra_om4_onedeg"
# NAME_short="Samudra_om4_onedeg"
# NAME_short2="Samudra_om4_onedeg_teacher"

# DATA_PATH=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg
# CKPT=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/$NAME_long/ema_ckpt.pt
# CONFIG_PATH=configs/samudra_om4/eval.yaml

# uv run --preview-features extra-build-dependencies -m ocean_emulators.eval \
#     $CONFIG_PATH \
#     --ckpt_path $CKPT \
#     --experiment.data_root $DATA_PATH \
#     --experiment.name $NAME_short2 \
#     --save_zarr true \
#     --forcing_mode teacher \
#     --experiment.wandb.mode disabled

# uv run --preview-features extra-build-dependencies -m ocean_emulators.viz configs/samudra_om4/viz.yaml \
#     --data_root $DATA_PATH \
#     --name "${NAME_short2}_cmp_viz" \
#     --runs="[{\"name\": \"$NAME_short\", \"location\": \"/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.LOCAL/${NAME_long}/predictions.zarr\"},{\"name\": \"$NAME_short2\", \"location\": \"/global/homes/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/.LOCAL/${NAME_short2}/predictions.zarr\"}]" 


NAME_long="2025-12-02-samudra_om4_v1_onedeg"
NAME_short="Samudra_om4_v1_onedeg"
NAME_short2="Samudra_om4_v1_onedeg_teacher"

DATA_PATH=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg
CONFIG_PATH=configs/samudra_om4_v1/eval.yaml
CKPT=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/$NAME_long/ema_ckpt.pt


uv run --preview-features extra-build-dependencies -m ocean_emulators.eval \
    $CONFIG_PATH \
    --ckpt_path $CKPT \
    --experiment.data_root $DATA_PATH \
    --experiment.name $NAME_short2 \
    --save_zarr true \
    --forcing_mode teacher \
    --experiment.wandb.mode disabled


NAME_long="2025-12-04-samudra_om4_v1_onedeg_blur"
NAME_short="Samudra_om4_v1_onedeg_blur"
NAME_short2="Samudra_om4_v1_onedeg_blur_teacher"

DATA_PATH=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/blur/om4_onedeg_blur_v8
CONFIG_PATH=configs/samudra_om4_v1/eval.yaml
CKPT=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/$NAME_long/ema_ckpt.pt


uv run --preview-features extra-build-dependencies -m ocean_emulators.eval \
    $CONFIG_PATH \
    --ckpt_path $CKPT \
    --experiment.data_root $DATA_PATH \
    --experiment.name $NAME_short2 \
    --save_zarr true \
    --forcing_mode teacher \
    --experiment.wandb.mode disabled

NAME_long="2025-12-04-samudra_om4_onedeg_blur"
NAME_short="Samudra_om4_onedeg_blur"
NAME_short2="Samudra_om4_onedeg_blur_teacher"

DATA_PATH=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/blur/om4_onedeg_blur_v8
CONFIG_PATH=configs/samudra_om4/eval.yaml
CKPT=/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/checkpoints/$NAME_long/ema_ckpt.pt


uv run --preview-features extra-build-dependencies -m ocean_emulators.eval \
    $CONFIG_PATH \
    --ckpt_path $CKPT \
    --experiment.data_root $DATA_PATH \
    --experiment.name $NAME_short2 \
    --save_zarr true \
    --forcing_mode teacher \
    --experiment.wandb.mode disabled

# sbatch -A m4874 -C gpu -q regular scripts/run_eval_onedeg.sbatch
# 2025-12-02-samudra_om4_halfdeg  2025-12-03-samudra_om4_onedeg