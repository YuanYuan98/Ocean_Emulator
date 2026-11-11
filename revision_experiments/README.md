# Samudra 2 revision experiments

Models trained for the Samudra 2 revision whose code is not part of the
repository's own `src/`. Each folder is self-contained: a full copy of the
`ocean_emulators` package exactly as that model was trained, the training
config as run, the small asset files it needs, and a README.

| Folder | Models | Trained |
|---|---|---|
| [`normalized_delta/`](normalized_delta/README.md) | normalized-delta (residual) model | 2026-10-29 to 2026-10-30 |
| [`loss_controlled/`](loss_controlled/README.md) | three loss controls: frozen final weights, tendency weights, Hu et al. adaptive weights | 2026-10-23 to 2026-10-26 |
| [`anomaly_background/`](anomaly_background/README.md) | anomaly-background model | 2026-10-29 to 2026-10-30 |
| [`deep_ocean_controls/`](deep_ocean_controls/README.md) | deep-ocean input controls: `all_ocean_input`, `deep_ocean_input` | 2026-11-02 to 2026-11-03 |

All are 1° OM4 models on the standard Samudra backbone, trained for 70 epochs
with seed 15 on Perlmutter.

## Why these are separate folders

These models were trained from code built on upstream commit `edd7b8c4`
(#558, 2026-01-29). The repository's own `src/` branched off earlier and uses a
different config format and loss interface, so the two cannot share one
package without rewriting one of them. Each folder therefore keeps its own
copy, and a model is run with `PYTHONPATH` pointing at that copy; the README
in each folder gives the command.

The folders are excluded from linting and type checking so the code stays
byte-identical to what ran.

## Other revision experiments

The forcing ablations, the earlier loss baselines, the Neptune baseline, and
the multi-start evaluations ran from the repository's own `src/`; see
`scripts/forcing_revision_*`, `scripts/revision_loss_*`, `scripts/multistart_*`
and `configs/neptune_om4/`.
