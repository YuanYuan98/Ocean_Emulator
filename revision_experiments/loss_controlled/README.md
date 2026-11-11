# Samudra 2 loss controls, 1° OM4

Code and configuration of the three loss-control models trained for the
Samudra 2 revision (2026-10-23 to 2026-10-26). All three use the standard 1°
Samudra backbone, data, and seed 15; only the per-channel loss weighting
differs from the reference model.

| Run | Config | Loss |
|---|---|---|
| `controlled_frozen_final` | `frozen_final.yaml` | fixed weights equal to the reference model's final dynamic weights (`frozen_final_physical_weights.json`) |
| `controlled_tendency` | `tendency.yaml`, then `tendency_shared2_local8_accum2_workers2.yaml` | fixed weights from the variance of the state increments (`tendency_physical_weights.json`) |
| `controlled_hu` | `hu.yaml` | adaptive weighting adapted from Hu et al. (`revision_hu`), no weight file |

## What is in this folder

`src/ocean_emulators/` is a full copy of the package exactly as it ran: upstream
commit `edd7b8c4` (#558, 2026-01-29) plus the files below, copied unchanged from
the runs' source snapshot. It is separate from the repository's own `src/`,
which is different code and does not know this model.

The files that differ from upstream are:

| File | Change |
|---|---|
| `src/ocean_emulators/utils/loss_controlled.py` | new: fixed-weight and Hu et al. adaptive losses |
| `src/ocean_emulators/config.py` | `revision_fixed` / `revision_hu` loss types, `offline` W&B mode |
| `src/ocean_emulators/train.py` | W&B logging enabled in offline mode |

## Running it

The configs hold absolute Perlmutter paths. Change these before use:

* `experiment.data_root`: directory with `OM4.zarr`, `OM4_means.zarr`,
  `OM4_stds.zarr` (1° data).
* `experiment.base_output_dir`
* `loss.weights_path` (not used by `hu.yaml`): point it to the JSON in this
  directory.

The entry point is the usual one, run from the repository root with
`PYTHONPATH` pointing at this folder's copy of the package. For example, on
4 GPUs of one node (the runs used 8 GPUs across 2 nodes):

```bash
PYTHONPATH=revision_experiments/loss_controlled/src \
python -m torch.distributed.run --nproc-per-node=4 \
    -m ocean_emulators.train revision_experiments/loss_controlled/hu.yaml
```

The runs themselves did not use this command directly. They were driven by a
wrapper that subclasses `Trainer` to stop and resume at epoch boundaries inside
4-hour allocations; the wrapper is not in this repository. The command above has
not been run end to end.

## How the runs were made

* Perlmutter, 70 epochs, effective global batch 32, seed 15.
* `controlled_frozen_final` and `controlled_hu`: 8 GPUs × batch 4.
* `controlled_tendency`: 8 GPUs × batch 4 with `tendency.yaml` through epoch
  50, then 2 GPUs × batch 8 with gradient accumulation 2
  (`tendency_shared2_local8_accum2_workers2.yaml`) for epochs 51 to 70.
* W&B: logged offline as runs `ekoaal4n` (frozen final), `rlsdej4p`
  (tendency), `36z7vif4` (Hu); they have not been synced, so there are no W&B
  pages.
* Outputs are on Perlmutter under
  `/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/experiments/loss_controlled_20261023/training/`.
