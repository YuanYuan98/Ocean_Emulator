# Observation Data Processing For Samudra Comparison

This note summarizes how the observation products were processed before being compared against `Samudra` and `Samudra 2` in `obs_evaluation.py`.

## Goal

The model outputs are:

- on the Samudra native horizontal grid: `180 x 360`
- approximately 1 degree resolution
- saved every 5 days

The observation products were not on the same temporal or spatial grid, so they were preprocessed to make the comparison better matched to the model output.

The preprocessing script is:

- [prepare_obs_5day_aligned.py](/global/u2/y/yy6080/workspace/Ocean_Emulator_Workspace/Ocean_Emulator/viz_jupyter/prepare_obs_5day_aligned.py)

The processed observation products currently used by `obs_evaluation.py` are:

- `/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/obs_data/modelgrid_1deg/duacs.zarr`
- `/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/obs_data/modelgrid_1deg/oisst.zarr`
- `/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/obs_data/modelgrid_1deg/argo-iap.zarr`

## Model Reference Grid

The processed obs products were matched to the `Samudra` rollout grid:

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
2. Apply a centered 5-sample rolling mean in time.
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

All three observation products were coarse-grained to the Samudra model grid.

The current method is:

1. Normalize longitude to `[0, 360)`.
2. Sort by latitude and longitude.
3. Coarsen the raw field by integer factors in latitude and longitude.
4. After coarse-graining, interpolate onto the exact Samudra latitude and longitude centers.

### Coarsening factors

- `DUACS`: from `1440 x 2880` to `180 x 360`
  so coarsening factor is `8 x 8`
- `OISST`: from `720 x 1440` to `180 x 360`
  so coarsening factor is `4 x 4`
- `ARGO-IAP`: from `360 x 720` to `180 x 360`
  so coarsening factor is `2 x 2`

After that step, the processed obs products live on the same horizontal grid as Samudra:

- `lat = 180`
- `lon = 360`
- longitude centers `0.5 ... 359.5`

## What Is Compared To What

### EKE map vs DUACS

Uses:

- DUACS on the processed Samudra horizontal grid
- DUACS on the 2022 aligned 5-day timestamps
- Samudra / Samudra 2 surface velocity at the same period

Metric:

- RMSE of time-mean surface EKE over 2022

### Isotropic temperature anomaly spectra vs OISST

Uses:

- OISST on the processed Samudra horizontal grid
- OISST on the aligned 5-day timestamps
- Samudra / Samudra 2 surface temperature over the same aligned period

Metric:

- RMSE between observation and model spectra, computed bucket-by-bucket after interpolation onto a common frequency axis

### Temporal KE spectra vs DUACS

Uses:

- DUACS on the processed Samudra horizontal grid
- DUACS on the aligned 5-day timestamps
- Samudra / Samudra 2 surface velocity over the same aligned period

Metric:

- RMSE between temporal KE spectra, bucket-by-bucket after interpolation onto a common frequency axis

### OHC monthly vs ARGO-IAP

Uses:

- ARGO-IAP on the processed Samudra horizontal grid
- ARGO-IAP monthly time axis
- Samudra / Samudra 2 monthly OHC derived from 5-day model output

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

### 2. The spatial coarsening is simple area-free block averaging

The current horizontal coarsening uses plain mean over integer grid blocks, then interpolation to the exact Samudra centers.

This is simple and reproducible, but it is not a conservative regridding scheme.

If stricter physical consistency is needed, a later refinement could replace this with conservative or area-weighted regridding.

### 3. ARGO-IAP depth range is limited

ARGO-IAP here only goes to `2000 m`, so only upper and intermediate OHC are comparable.

### 4. Spectral RMSE is not log-space RMSE

The current spectrum RMSE is computed directly on the spectral values after interpolation to a common spectral axis. If a log-space comparison is preferred, that would be a methodological change.

## Why This Processing Was Chosen

The main design choices were:

- compare on the Samudra horizontal grid rather than the finer obs grid
- compare DUACS and OISST on the Samudra 5-day timeline
- keep ARGO-IAP monthly because monthly OHC is the intended target

This makes the obs products closer to what the emulator can actually represent, while avoiding introducing fake high-frequency variability into the monthly OHC comparison.

## Current Default In Code

`obs_evaluation.py` now assumes:

- `DUACS` and `OISST` come from `modelgrid_1deg`
- `ARGO-IAP` also comes from `modelgrid_1deg`, but still monthly

So running

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python obs_evaluation.py
```

will use the processed obs products described above.
