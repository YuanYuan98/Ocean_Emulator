# Observation Data Processing For Samudra Comparison

This note summarizes how the observation products were processed before being compared against `Samudra` and `Samudra 2` in `obs_evaluation.py`.

## Goal

The model outputs are:

- on the Samudra native horizontal grid: `180 x 360`
- approximately 1 degree resolution
- saved every 5 days

The observation products were not on the same temporal grid, so they were preprocessed in time to match the rollout cadence. Spatially, each observation product is **kept on its own native grid** and compared against the model on the model's own grid (no spatial resampling is done at preprocessing time).

The preprocessing script is:

- [prepare_obs_5day_aligned.py](/global/u2/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/viz_jupyter/prepare_obs_5day_aligned.py)

The processed observation products currently used by `obs_evaluation.py` are:

- `/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/obs_data/nativegrid/duacs.zarr`
- `/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/obs_data/nativegrid/oisst.zarr`
- `/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/obs_data/nativegrid/argo-iap.zarr`

## Model Reference Grid

The temporal alignment is matched to the `Samudra` rollout grid:

- latitude size: `180`
- longitude size: `360`
- latitude centers start near `-89.2366` and end near `89.2366`
- longitude centers are `0.5, 1.5, ..., 359.5`

The 2022 rollout time axis used as the main reference is:

- start: `2022-01-03 12:00:00`
- end: `2022-12-24 12:00:00`
- cadence: every `5` days
- number of samples: `72`

## Raw Observation Products

The raw products had different native grids:

- `DUACS`: daily, about `0.125 degree`
- `OISST`: daily, about `0.25 degree`
- `ARGO-IAP`: monthly, about `0.5 degree`, depth only to `2000 m`

## Time Processing

### DUACS

Raw DUACS is daily.

To compare with Samudra:

1. Read the daily 2022 DUACS fields.
2. Apply a centered 5-sample rolling mean in time (`min_periods=5`).
3. Sample those rolling means on the Samudra 2022 rollout dates.
4. Relabel the result to the exact Samudra timestamps.

So the final DUACS comparison product is:

- temporally aligned to the model
- represented on the same 72 timestamps as the rollout

### OISST

Raw OISST is daily.

The same time processing as DUACS was used:

1. Read daily 2022 OISST.
2. Apply a centered 5-sample rolling mean.
3. Sample on the Samudra rollout dates.
4. Relabel to the exact rollout timestamps.

So the final OISST comparison product also has:

- `72` timestamps
- exact match to the Samudra 2022 5-day timeline

### ARGO-IAP

Raw ARGO-IAP is monthly, not daily.

Because of that, it was **not** forced onto a 5-day timeline in the final setup.

Instead:

- it stays monthly in time
- it is used for monthly OHC comparisons only

This is important because converting monthly ARGO-IAP into 5-day values would introduce artificial time information rather than real observations.

## Horizontal Spatial Processing

There is no spatial coarsening or interpolation in `prepare_obs_5day_aligned.py`. The processed obs products keep the source product's native horizontal resolution:

- `DUACS`: ~`0.125 degree`
- `OISST`: ~`0.25 degree`
- `ARGO-IAP`: ~`0.5 degree` (monthly, native grid)

The only spatial operations applied are reproducible coordinate housekeeping:

1. Rename `latitude`/`longitude` to `lat`/`lon` if needed.
2. Normalize longitude to `[0, 360)` and sort by longitude when the source uses `[-180, 180)`.
3. Sort by latitude.

`obs_evaluation.py` then computes diagnostics independently on each side's own grid (the obs grid for obs, the model grid for the model) and compares the resulting quantities (maps, spectra, time series). This avoids ambiguity about which regridding scheme is used and pushes the question of "what scale is fair to compare" into the diagnostic itself rather than into the preprocessed file.

## What Is Compared To What

### EKE map vs DUACS

Uses:

- DUACS on its native ~0.125° grid, on the 2022 aligned 5-day timestamps
- Samudra / Samudra 2 surface geostrophic velocity (derived from `zos`) on the model grid, same period

Each side's EKE map is plotted on its own grid, then a global area-weighted mean is reported per side.

Metric:

- global mean of time-mean surface geostrophic EKE over 2022 (and the bias against DUACS)

### Isotropic temperature anomaly spectra vs OISST

Uses:

- OISST on its native ~0.25° grid, on the aligned 5-day timestamps
- Samudra / Samudra 2 surface temperature on the model grid, same aligned period

Each side's regional 2D spectrum is computed on its own grid using its own `dx`/`dy`. The two curves are then compared bucket-by-bucket after interpolation onto a common wavenumber axis.

Metric:

- RMSE between observation and model isotropic spectra

### Temporal KE spectra vs DUACS

Uses:

- DUACS on its native ~0.125° grid, on the aligned 5-day timestamps
- Samudra / Samudra 2 surface geostrophic velocity on the model grid, same aligned period

Each side's regional temporal KE spectrum (Welch PSD over time, averaged over the region) is computed on its own grid. The two curves are compared bucket-by-bucket after interpolation onto a common frequency axis.

Metric:

- RMSE between temporal KE spectra

### OHC monthly vs ARGO-IAP

Uses:

- ARGO-IAP on its native monthly, ~0.5° grid (grid metrics inferred at evaluation time)
- Samudra / Samudra 2 monthly OHC, derived by averaging the 5-day model output to monthly

Each side integrates per-layer OHC on its own grid, so no spatial regridding is required.

Only the following layers are currently used:

- `0-700 m`
- `700-2000 m`

The `2000-7000 m` layer is **not** used now, because the ARGO-IAP product only extends to `2000 m`.

Metrics:

- monthly OHC RMSE
- monthly OHC anomaly RMSE

## Important Caveats

### 1. The temporal filtering is different across products

- DUACS and OISST are explicitly converted to a 5-day representation
- ARGO-IAP remains monthly

This is intentional:

- high-frequency products are matched to the 5-day model cadence
- monthly OHC comparisons stay monthly

### 2. No spatial regridding is done at preprocessing time

Each side stays on its own grid through the diagnostic. This means:

- map comparisons are visualized on different native grids and only summary statistics (e.g. global means) are directly compared as numbers
- spectra are computed on each side's own grid and reconciled in spectral space, not in physical space
- there is no area-weighted or conservative regridding step that could be a source of methodological error

If a future diagnostic needs the obs and model on a common grid (e.g. for a pointwise difference map), that regridding has to be added explicitly in `obs_evaluation.py`; the preprocessed files will not do it for you.

### 3. ARGO-IAP depth range is limited

ARGO-IAP here only goes to `2000 m`, so only upper and intermediate OHC are comparable.

### 4. Spectral RMSE is not log-space RMSE

The current spectrum RMSE is computed directly on the spectral values after interpolation to a common spectral axis. If a log-space comparison is preferred, that would be a methodological change.

## Why This Processing Was Chosen

The main design choices were:

- align DUACS and OISST onto the Samudra 5-day timeline so that map and spectral diagnostics see the same temporal sampling on both sides
- keep ARGO-IAP monthly because monthly OHC is the intended target
- keep each obs product on its native horizontal grid and let the diagnostic decide what is fair to compare, rather than baking a particular regridding choice into the preprocessed files

This avoids introducing fake high-frequency variability into the monthly OHC comparison, and keeps the preprocessing step minimal and reproducible.

## Current Default In Code

`obs_evaluation.py` now assumes:

- `DUACS`, `OISST`, and `ARGO-IAP` all come from `nativegrid/`
- DUACS and OISST are on the 5-day aligned timeline; ARGO-IAP is still monthly

So running

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python obs_evaluation.py
```

will use the processed obs products described above. The default location can be overridden via `--aligned-obs-root`.
