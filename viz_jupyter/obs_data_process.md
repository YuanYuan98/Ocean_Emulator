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
- `/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/obs_data/nativegrid/oisst_daily.zarr`
- `/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/obs_data/nativegrid/argo-iap.zarr`

The prepared ARGO store now includes both `temp` and `salt`, so
temperature/salinity spatial-vertical structure diagnostics can run directly
against `nativegrid/argo-iap.zarr`.

Model-native diagnostics in `obs_evaluation.py` use the OM4 finite ocean mask
as the common model wet mask for both OM4 and Samudra / Samudra 2. This avoids
treating finite Samudra prediction values over land or otherwise invalid cells
as valid ocean values.

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

The long OISST spectrum comparison now also uses the Samudra 2 overlap
timeline:

- start: `2014-10-20 12:00:00`
- end: `2022-12-24 12:00:00`
- cadence: model rollout timestamps, mostly every `5` days with a few `6` day
  gaps inherited from the rollout calendar
- number of 5-day samples: `598`

## Raw Observation Products

The raw products had different native grids:

- `DUACS`: daily, about `0.125 degree`
- `OISST`: daily, about `0.25 degree`; the local raw copy covers
  `1982-01-01` through `2022-12-31`
- `ARGO-IAP`: monthly, about `0.5 degree`, depth only to `2000 m`; the local
  raw copy contains matching 1980-2022 temperature and salinity files.

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

Two prepared OISST products are now written:

- `nativegrid/oisst.zarr`: centered 5-day rolling mean sampled on the Samudra 2
  rollout timestamps from `2014-10-20 12:00:00` through
  `2022-12-24 12:00:00` (`598` samples)
- `nativegrid/oisst_daily.zarr`: raw daily OISST on the same native grid from
  `2014-10-20 12:00:00` through `2022-12-24 12:00:00` (`2988` samples)

The 5-day processing is:

1. Read daily OISST over the requested comparison window plus the two-day
   centered-rolling margin.
2. Apply a centered 5-sample rolling mean.
3. Sample on the Samudra 2 rollout dates.
4. Relabel to the exact rollout timestamps.

The daily product is standardized in coordinates only; it is not temporally
smoothed.

### ARGO-IAP

Raw ARGO-IAP is monthly, not daily.

Because of that, it was **not** forced onto a 5-day timeline in the final setup.

Instead:

- it stays monthly in time
- it is used for monthly OHC, temperature, and salinity time-series
  comparisons over the common model/ARGO overlap
- its full 1980-2022 monthly record is used as the observational seasonal cycle
  for monthly climatology diagnostics
- it is used for annual-mean temperature/salinity/OHC map and structure
  comparisons

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

- raw daily OISST on its native ~0.25° grid
- 5-day averaged OISST on its native ~0.25° grid, sampled on the aligned
  Samudra 2 timestamps
- Samudra / Samudra 2 surface temperature on the model grid, same aligned period

Each side's regional 2D spectrum is computed on its own grid using its own
`dx`/`dy`. The raw daily OISST, 5-day averaged OISST, and model curves are
plotted together. Model RMSE is reported separately against raw daily OISST and
against 5-day averaged OISST after interpolation onto a common wavenumber axis.

Metric:

- RMSE between model and raw daily OISST isotropic temperature spectra
- RMSE between model and 5-day averaged OISST isotropic temperature spectra

Metric:

- RMSE between observation and model isotropic spectra

### Temporal KE spectra vs DUACS

Uses:

- raw daily DUACS on its native ~0.125° grid
- 5-day averaged DUACS on the aligned Samudra timestamps
- Samudra / Samudra 2 surface geostrophic velocity on the model grid, same aligned period

Each side's regional temporal KE spectrum (Welch PSD over time, averaged over the region) is computed on its own grid. The raw daily DUACS, 5-day averaged DUACS, and model curves are plotted together. Model RMSE is reported separately against raw daily DUACS and against 5-day averaged DUACS, using only overlapping frequency support.

Metric:

- RMSE between model and raw daily DUACS temporal KE spectra
- RMSE between model and 5-day averaged DUACS temporal KE spectra

### OHC monthly vs ARGO-IAP

Uses:

- ARGO-IAP on its native monthly, ~0.5° grid
- Samudra / Samudra 2 monthly OHC, derived by averaging the 5-day model output
  to monthly

Each side integrates per-layer OHC on its own grid, so no spatial regridding is
required.

Only the following layers are currently used:

- `0-700 m`
- `700-2000 m`

The `2000-7000 m` layer is not used because the ARGO-IAP product only extends
to `2000 m`.

Metrics:

- monthly OHC RMSE
- monthly OHC anomaly RMSE
- monthly OHC climatology RMSE

### Temperature/salinity monthly time series vs ARGO-IAP

Uses:

- ARGO-IAP monthly temperature and salinity on its native ~0.5° grid
- Samudra / Samudra 2 monthly layer-mean `thetao` and `so`, derived by
  averaging the 5-day model output to monthly

Each side computes global volume-weighted layer means on its own grid. The
layers match the OHC diagnostics:

- `0-700 m`
- `700-2000 m`

Metrics:

- monthly temperature layer-mean RMSE
- monthly temperature layer-mean anomaly RMSE
- monthly temperature layer-mean climatology RMSE
- monthly salinity layer-mean RMSE
- monthly salinity layer-mean anomaly RMSE
- monthly salinity layer-mean climatology RMSE

### Temperature/salinity/OHC maps vs ARGO-IAP

Uses:

- ARGO-IAP annual-mean temperature and salinity on its native ~0.5° grid
- Samudra / Samudra 2 annual-mean `thetao` and `so`, interpolated onto the
  ARGO latitude, longitude, and depth grid

Temperature and salinity maps show volume-weighted layer means for `0-700 m`
and `700-2000 m`. OHC maps show layer-integrated heat content per unit area in
`J m^-2`, computed from temperature, layer thickness, seawater density, and heat
capacity. After interpolation, model fields are masked to the ARGO-IAP finite
ocean mask before plotting or computing map diagnostics. Map color scales use
the ARGO-IAP robust range; model values outside that range are shown with
colorbar extensions.

Metrics:

- annual-mean temperature layer-map RMSE and mean bias
- annual-mean salinity layer-map RMSE and mean bias
- annual-mean OHC per-area layer-map RMSE and mean bias

### Temperature/salinity/OHC structure vs ARGO-IAP

Uses:

- ARGO-IAP monthly temperature and salinity on its native ~0.5° grid
- Samudra / Samudra 2 annual-mean `thetao` and `so`, interpolated onto the
  ARGO latitude, longitude, and depth grid inside `obs_evaluation.py`

Unlike the other native-grid diagnostics, this comparison performs explicit
pointwise interpolation because the goal is to compare spatial and vertical
structure. After interpolation, model fields are masked to the ARGO-IAP finite
ocean mask before zonal means are computed.

Diagnostic:

- separate zonal-mean annual-mean distribution figures for ARGO-IAP, Samudra 2,
  and OM4 for temperature, salinity, and per-layer OHC
- separate zonal-mean annual-mean model-minus-ARGO bias figures for
  temperature, salinity, and per-layer OHC versus latitude and depth

Metric:

- area- and layer-thickness-weighted spatial/vertical RMSE for annual-mean
  temperature and salinity over the common 0-2000 m ARGO depth range
- area-weighted spatial/vertical RMSE for per-layer OHC bias in `J m^-2`

## Important Caveats

### 1. The temporal filtering is different across products

- DUACS and OISST are explicitly converted to a 5-day representation
- raw daily DUACS is loaded directly for temporal-frequency comparisons
- raw daily OISST is also prepared for daily-vs-5-day SST spectrum comparisons
- ARGO-IAP remains monthly

This is intentional:

- high-frequency products are matched to the 5-day model cadence where the
  model is directly sampled
- raw daily observation spectra are kept as additional references for the
  effect of temporal smoothing
- monthly OHC comparisons stay monthly
- ARGO temperature/salinity structure comparisons stay monthly and are then
  summarized as annual means

### 2. No spatial regridding is done at preprocessing time

Each side stays on its own grid through the diagnostic. This means:

- map comparisons are visualized on different native grids and only summary statistics (e.g. global means) are directly compared as numbers
- spectra are computed on each side's own grid and reconciled in spectral space, not in physical space
- there is no area-weighted or conservative regridding step that could be a source of methodological error

The ARGO temperature/salinity structure diagnostic is the exception: it
interpolates the annual-mean model fields onto the ARGO grid inside
`obs_evaluation.py`, because pointwise spatial and vertical structure is the
quantity being evaluated. That interpolation is deliberately not baked into the
preprocessed files.

### 3. ARGO-IAP monthly climatology is observation-based

The current local ARGO-IAP raw and prepared products cover January 1980 through
December 2022. The default monthly climatology calculation uses the full
prepared 1980-2022 ARGO-IAP record. This gives a stable observational seasonal
cycle for the monthly climatology figures even when a model comparison window is
shorter than the full ARGO record.

### 4. ARGO-IAP depth range is limited

ARGO-IAP here only goes to `2000 m`, so temperature/salinity structure is only
compared over the common ARGO-covered depth range.

### 5. Spectral RMSE is not log-space RMSE

The current spectrum RMSE is computed directly on the spectral values after interpolation to a common spectral axis. If a log-space comparison is preferred, that would be a methodological change.

## Why This Processing Was Chosen

The main design choices were:

- align DUACS and OISST onto the Samudra 5-day timeline so that map and some
  spectral diagnostics see the same temporal sampling on both sides
- keep raw daily OISST and raw daily DUACS available for spectrum diagnostics
  that explicitly compare daily and 5-day observational sampling
- keep ARGO-IAP monthly because OHC and temperature/salinity structure are
  monthly observational products
- keep each obs product on its native horizontal grid and let the diagnostic decide what is fair to compare, rather than baking a particular regridding choice into the preprocessed files

This avoids introducing fake high-frequency variability into ARGO comparisons,
including OHC, and keeps the preprocessing step minimal and reproducible.

## Current Default In Code

`obs_evaluation.py` now assumes:

- `DUACS`, `OISST`, and `ARGO-IAP` all come from `nativegrid/`
- DUACS and `oisst.zarr` are on the 5-day aligned timeline; ARGO-IAP is still
  monthly
- `oisst_daily.zarr` is used as the raw-daily OISST reference for SST spectra
- raw daily DUACS is also loaded from `--raw-obs-root` for temporal spectra
- OHC uses the common ARGO/model overlap by default; `--ohc-start-date` and
  `--ohc-end-date` can force a specific monthly OHC window when longer ARGO
  products are available
- ARGO-IAP monthly climatologies default to the full prepared ARGO time range;
  `--obs-climatology-start-date` and `--obs-climatology-end-date` can override
  the climatology window
- the default DUACS velocity kind is `absolute`, meaning `ugos`/`vgos` from
  absolute dynamic topography; `--duacs-velocity-kind anomaly` switches to
  `ugosa`/`vgosa` from sea-level anomaly and uses model geostrophic velocity
  anomalies for the temporal spectra

So running

```bash
../.venv/bin/python obs_evaluation.py
```

will use the processed obs products described above. The default location can be overridden via `--aligned-obs-root`.
