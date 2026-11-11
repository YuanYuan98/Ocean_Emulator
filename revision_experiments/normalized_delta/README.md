# Samudra 2 normalized-delta (residual) model, 1° OM4

Code and configuration of the normalized-delta residual model trained for the
Samudra 2 revision (run `normalized_delta`, 2026-10-29 to 2026-10-30).

The network is the standard 1° Samudra backbone. Instead of the next two
states, its two output heads predict per-channel normalized increments: head 1
is the increment from the last input state, head 2 the increment from the
head-1 prediction. The loss is an unweighted masked MSE on those normalized
increments.

## What is in this folder

`src/ocean_emulators/` is a full copy of the package exactly as it ran: upstream
commit `edd7b8c4` (#558, 2026-01-29) plus the files below, copied unchanged from
the run's source snapshot. It is separate from the repository's own `src/`,
which is different code and does not know this model.

The files that differ from upstream are:

| File | Change |
|---|---|
| `src/ocean_emulators/models/normalized_delta.py` | new: increment targets and decoding |
| `src/ocean_emulators/utils/loss_normalized_delta.py` | new: loss on normalized increments |
| `src/ocean_emulators/utils/loss_controlled.py` | new: fixed-weight loss base class |
| `src/ocean_emulators/config.py` | `normalized_delta` loss type, statistics path, `offline` W&B mode |
| `src/ocean_emulators/models/base.py` | decode increments back to states |
| `src/ocean_emulators/stepper.py` | validation on increments |
| `src/ocean_emulators/train.py` | per-channel increment metrics, EMA and non-EMA validation |

Configuration, in this folder:

* `train.yaml`: the training config exactly as run.
* `normalized_delta_statistics.json`: per-channel increment scales fitted on
  the training period (sha256 `e656743a…`, checked by the code on resume).

## Running it

`train.yaml` holds absolute Perlmutter paths. Change these four before use:

* `experiment.data_root`: directory with `OM4.zarr`, `OM4_means.zarr`,
  `OM4_stds.zarr` (1° data).
* `experiment.base_output_dir`
* `loss.weights_path` and `model.normalized_delta_statistics_path`: both point
  to `normalized_delta_statistics.json`.

The entry point is the usual one, run from the repository root with
`PYTHONPATH` pointing at this folder's copy of the package:

```bash
PYTHONPATH=revision_experiments/normalized_delta/src \
python -m torch.distributed.run --nnodes=1 --nproc-per-node=4 \
    -m ocean_emulators.train revision_experiments/normalized_delta/train.yaml
```

The run itself did not use this command directly. It was driven by a wrapper
that subclasses `Trainer` to stop and resume at epoch boundaries inside 4-hour
interactive allocations; the wrapper is not in this repository. The command above
has not been run end to end.

## How the run was made

* Perlmutter, one GPU node, 4 GPUs, `interactive` QOS, 440 GB host memory.
* 4 ranks × batch 8 (global batch 32), 70 epochs, 6160 optimizer updates,
  seed 15, learning rate 6e-4 with cosine schedule, EMA 0.999.
* Environment: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  `WANDB_MODE=offline`, `OMP_NUM_THREADS=8`.
* W&B: logged offline as run `4mjcaewt`; it has not been synced, so there is
  no W&B page.
* Outputs (checkpoints, logs, offline W&B files) are on Perlmutter under
  `/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/revision-boss/normalized-delta-experiment-20261029/training_validation_device_v4/normalized_delta/`.
