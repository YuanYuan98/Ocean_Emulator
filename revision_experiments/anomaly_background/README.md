# Samudra 2 anomaly-background model, 1° OM4

Code and configuration of the anomaly-background model trained for the
Samudra 2 revision (run `anomaly_background`, 2026-10-29 to 2026-10-30).

The network is the standard 1° Samudra backbone. A fixed background field is
fitted per channel and grid cell on the training period (mean, linear trend,
annual and semiannual harmonics). The two output heads predict the normalized
anomaly from that background at each target date, and the prediction is
decoded as background plus anomaly.

## What is in this folder

`src/ocean_emulators/` is a full copy of the package exactly as it ran: upstream
commit `edd7b8c4` (#558, 2026-01-29) plus the files below, copied unchanged from
the run's source snapshot. It is separate from the repository's own `src/`,
which is different code and does not know this model.

The same code as a diff against upstream is on branch `2026-10-29/expt/anomaly-background`.

| File | Change |
|---|---|
| `src/ocean_emulators/models/anomaly_background.py` | new: background evaluation, anomaly targets, decoding |
| `src/ocean_emulators/utils/anomaly_time.py` | new: date features used to evaluate the background |
| `src/ocean_emulators/utils/loss_anomaly_background.py` | new: loss on normalized anomalies |
| `src/ocean_emulators/models/normalized_delta.py`, `utils/loss_normalized_delta.py`, `utils/loss_controlled.py` | carried over from the normalized-delta snapshot this one was copied from |
| `src/ocean_emulators/config.py`, `datasets.py`, `models/base.py`, `stepper.py`, `train.py`, `utils/train.py` | wiring: loss type, statistics path, dates passed with each batch |

Configuration, in this folder:

* `train.yaml`: the training config exactly as run.
* `background_specification.json`: the fitted background's definition and
  per-channel statistics.
* `fit_background.py`: the script that fitted the background from the 1° OM4
  training data. It is a copy for reference: it was run from the experiment's
  `checks/` directory and has hard-coded Perlmutter paths.

The fitted background fields themselves (77 `.npy` files, about 117 MB) are
not in git. They are on Perlmutter next to the original specification, under
`/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/revision-boss/anomaly-background-experiment-20261029/assets/background_fit_full/`,
and the model needs them at that location or regenerated with
`fit_background.py`.

## Running it

`train.yaml` holds absolute Perlmutter paths. Change these before use:

* `experiment.data_root`: directory with `OM4.zarr`, `OM4_means.zarr`,
  `OM4_stds.zarr` (1° data).
* `experiment.base_output_dir`
* `loss.weights_path` and `model.anomaly_background_statistics_path`: both
  point to `background_specification.json`, which must sit beside the `.npy`
  background fields.

The entry point is the usual one, run from the repository root with
`PYTHONPATH` pointing at this folder's copy of the package:

```bash
PYTHONPATH=revision_experiments/anomaly_background/src \
python -m torch.distributed.run --nnodes=1 --nproc-per-node=4 \
    -m ocean_emulators.train revision_experiments/anomaly_background/train.yaml
```

The run itself did not use this command directly. It was driven by a wrapper
that subclasses `Trainer` to stop and resume at epoch boundaries inside 4-hour
interactive allocations; the wrapper is not in this repository. The command above
has not been run end to end.

## How the run was made

* Perlmutter, one GPU node, 4 GPUs × batch 8 (global batch 32), 70 epochs,
  seed 15, learning rate 6e-4 with cosine schedule, EMA 0.999.
* W&B: logged offline as run `bpjuxqty`; it has not been synced, so there is
  no W&B page.
* Outputs are on Perlmutter under
  `/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/revision-boss/anomaly-background-experiment-20261029/training_validation_device_v4/anomaly_background/`.
