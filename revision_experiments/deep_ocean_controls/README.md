# Samudra 2 deep-ocean input controls, 1° OM4

Code and configuration of the two deep-ocean control models trained for the
Samudra 2 revision (2026-11-02 to 2026-11-03). They test whether ocean state
above the deep ocean contributes to deep temperature and salinity artifacts.

Both use the standard 1° Samudra backbone with unchanged tensor shapes. Only
deep temperature and salinity (the five layers centred at 2400 m and below,
10 channels) are supervised, with the original dynamic loss restricted to
those channels.

| Run | Config | Ocean-state inputs |
|---|---|---|
| `all_ocean_input` | `all_ocean_input.yaml` | all original inputs; non-deep inputs come from the true state at each step |
| `deep_ocean_input` | `deep_ocean_input.yaml` | non-deep inputs masked to normalized zero at every step |

Atmospheric forcing inputs are kept in both. Outputs for channels outside the
10 deep ones are not trained and are not usable predictions.

## What is in this folder

`src/ocean_emulators/` is a full copy of the package exactly as it ran: upstream
commit `edd7b8c4` (#558, 2026-01-29) plus the files below, copied unchanged from
the runs' source snapshot. It is separate from the repository's own `src/`,
which is different code and does not know this model.

The same code as a diff against upstream is on branch `2026-11-02/expt/deep-ocean-controls`.

| File | Change |
|---|---|
| `src/ocean_emulators/models/deep_ocean_control.py` | new: input masking and deep-only feedback |
| `src/ocean_emulators/utils/loss_deep_ocean.py` | new: dynamic loss on the deep channels |
| `src/ocean_emulators/models/normalized_delta.py`, `utils/loss_normalized_delta.py`, `utils/loss_controlled.py` | carried over from the normalized-delta snapshot this one was copied from |
| `src/ocean_emulators/config.py`, `models/base.py`, `models/samudra.py`, `stepper.py`, `train.py` | wiring: `deep_dynamic` loss type, control mode, contract path |

Configuration, in this folder:

* `all_ocean_input.yaml`, `deep_ocean_input.yaml`: the training configs
  exactly as run. They differ only in `experiment.name` and
  `model.deep_ocean_control_mode`.
* `deep_control_contract.json`: the deep channel names and indices the code
  checks against.

## Running it

The configs hold absolute Perlmutter paths. Change these before use:

* `experiment.data_root`: directory with `OM4.zarr`, `OM4_means.zarr`,
  `OM4_stds.zarr` (1° data).
* `experiment.base_output_dir`
* `model.deep_ocean_control_contract_path`: point it to
  `deep_control_contract.json`.

The entry point is the usual one, run from the repository root with
`PYTHONPATH` pointing at this folder's copy of the package:

```bash
PYTHONPATH=revision_experiments/deep_ocean_controls/src \
python -m torch.distributed.run --nnodes=1 --nproc-per-node=4 \
    -m ocean_emulators.train revision_experiments/deep_ocean_controls/deep_ocean_input.yaml
```

The runs themselves did not use this command directly. They were driven by a
wrapper that subclasses `Trainer` to stop and resume at epoch boundaries inside
4-hour interactive allocations; the wrapper is not in this repository. The command
above has not been run end to end.

## How the runs were made

* Perlmutter, one GPU node each, 4 GPUs × batch 8 (global batch 32), 70
  epochs, seed 15.
* W&B: logged offline as runs `8c639drx` (`all_ocean_input`) and `uf6wemcu`
  (`deep_ocean_input`); they have not been synced, so there are no W&B pages.
* Outputs are on Perlmutter under
  `/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/revision-boss/deep-ocean-controls-20261102/training/`.
