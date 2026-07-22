# Observation Evaluation Metrics Reference

This note documents the current `obs_evaluation.py` observation-data evaluation workflow.
It is intended as a quick reference for future agents that need to reuse or extend these
metrics without reverse-engineering the script.

## Source Script And Main Outputs

- Main script: `obs_evaluation.py`
- Main metrics CSV: `Figures_observations/pdf_figures/metrics/observation_metrics_<year>.csv`
- Default RMSE/map year: `--year 2022`
- Current whole-rollout overlap in generated figures: approximately `2014-10` to `2022-12`
- Main observation products:
  - DUACS: surface geostrophic velocity and EKE comparisons
  - OISST: SST comparisons
  - ARGO-IAP: OHC comparisons
- Model products:
  - Samudra 2 and OM4 are reduced on their native grids where possible; fields are interpolated
    to observation grids when a pointwise map comparison requires paired cells.

All scalar map reductions use finite paired ocean cells only. Area weighting uses the available
grid-cell area, or an inferred spherical grid area for observation products when needed.

## Common Notation

Let:

- \(M_i(t)\) be the model value at grid cell \(i\) and time \(t\).
- \(O_i(t)\) be the observation value at grid cell \(i\) and time \(t\).
- \(A_i\) be the grid-cell area.
- \(e_i(t)=M_i(t)-O_i(t)\).
- `finite` means the model and observation values are both finite after masking and alignment.

Area-weighted mean:

\[
\langle X\rangle_A = \frac{\sum_{i \in finite} A_i X_i}{\sum_{i \in finite} A_i}
\]

Area-weighted map RMSE:

\[
RMSE_A(X) = \sqrt{\langle X^2\rangle_A}
\]

Time-series RMSE on the common timestamps:

\[
RMSE_t = \sqrt{\frac{1}{N}\sum_{t \in common} (M(t)-O(t))^2}
\]

Log10 curve RMSE for spectra:

1. Keep only finite, strictly positive x and PSD values over the common x-domain.
2. Interpolate the model log10 PSD onto the observation x-axis in log10-x space.
3. Compute the RMS log10 PSD difference:

\[
RMSE_{\log_{10}} =
\sqrt{\frac{1}{N}\sum_k \left(\log_{10}P_M(k)-\log_{10}P_O(k)\right)^2}
\]

The result is in dex and measures multiplicative rather than absolute PSD error.

## Region Definitions For Spectra

The spectrum panels are:

| Region | Longitude slice | Latitude slice |
| --- | --- | --- |
| Global | 0 to 360 | -90 to 90 |
| Gulf Stream | 300 to 320 | 25 to 45 |
| Agulhas | 40 to 60 | -50 to -30 |

The plotted spectrum figures use a 1 x 3 layout in this order:
Global, Gulf Stream, Agulhas.

## 1. Total RMSE + Spatial RMSE Maps

This block uses the requested evaluation year, default `2022`. Each product is first restricted
to the common overlap among observations and model outputs inside that year.

### 1.1 Velocity Vector RMSE

Observation: DUACS surface geostrophic velocity.

Model: geostrophic velocity derived from model SSH/zos on the observation grid.

The script masks the equatorial band \(|lat| < 5^\circ\) because geostrophic velocity is ill
conditioned there.

For each model and each observation grid cell:

\[
RMSE_i =
\sqrt{\frac{1}{T}\sum_t \left[(u_M(t)-u_O(t))^2 + (v_M(t)-v_O(t))^2\right]}
\]

The figure shows this spatial RMSE map. The scalar total RMSE is:

\[
RMSE_{total} = \sqrt{\langle RMSE_i^2\rangle_A}
\]

Metric key:

- `surface_geostrophic_velocity_vector_total_rmse`

Figure:

- `Figures_observations/pdf_figures/rmse/surface_geostrophic_velocity_vector_rmse_vs_duacs_<year>.pdf`

### 1.2 EKE RMSE

EKE is computed from velocity anomalies relative to each product's time mean:

\[
u'(t)=u(t)-\overline{u}, \quad v'(t)=v(t)-\overline{v}
\]

\[
EKE_i = \frac{1}{2}\overline{u_i'(t)^2 + v_i'(t)^2}
\]

The spatial EKE RMSE map is currently the absolute map difference:

\[
RMSE_i = |EKE_{M,i} - EKE_{O,i}|
\]

Because each EKE map is already a time-mean quantity, this is not a time-RMSE of instantaneous
EKE. The total RMSE is the area-weighted RMS of that difference map:

\[
RMSE_{total} = \sqrt{\langle (EKE_M-EKE_O)^2\rangle_A}
\]

Metric key:

- `surface_eke_total_rmse`

Figure:

- `Figures_observations/pdf_figures/rmse/surface_eke_rmse_vs_duacs_<year>.pdf`

### 1.3 SST RMSE

Observation: OISST.

Model: model surface temperature `thetao` at the top level, interpolated to the OISST grid.

For each grid cell:

\[
RMSE_i = \sqrt{\frac{1}{T}\sum_t (SST_M(t)-SST_O(t))^2}
\]

The scalar total RMSE is:

\[
RMSE_{total} = \sqrt{\langle RMSE_i^2\rangle_A}
\]

Metric key:

- `surface_sst_total_rmse`

Figure:

- `Figures_observations/pdf_figures/rmse/sst_rmse_maps_vs_oisst_<year>.pdf`

### 1.4 OHC RMSE: Upper 700 m And 700-2000 m

Observation: ARGO-IAP temperature.

Model: model temperature interpolated to the ARGO-IAP grid and common depth levels.

OHC per unit area for a layer \(z_1:z_2\):

\[
OHC_{area}(x,y,t) =
\rho_0 C_p \sum_{z=z_1}^{z_2} T(x,y,z,t)\Delta z
\]

The script uses:

- \(\rho_0=1035\) kg m\(^{-3}\)
- \(C_p=3850\) J kg\(^{-1}\) K\(^{-1}\)
- \(\rho_0 C_p=3{,}984{,}750\) J m\(^{-3}\) K\(^{-1}\)
- OHC reference: temperature integral relative to 0 °C, not a published IAP anomaly baseline
- `0-700 m` (`Surface`) and `700-2000 m` (`Intermediate`) layers

These values and the reference convention are written into the OHC metric rows. Published IAP
OHC values require conversion when their volumetric heat capacity or anomaly reference differs.

OHC fields are monthly averaged before RMSE. For each grid cell and layer:

\[
RMSE_i = \sqrt{\frac{1}{T}\sum_t
\left(OHC_{area,M}(t)-OHC_{area,O}(t)\right)^2}
\]

The total RMSE is:

\[
RMSE_{total} = \sqrt{\langle RMSE_i^2\rangle_A}
\]

Metric key:

- `ohc_per_area_total_rmse`, with `depth` equal to `Surface (0-700m)` or `Intermediate (700-2000m)`

Figure:

- `Figures_observations/pdf_figures/rmse/ohc_per_area_rmse_maps_vs_argo_iap_<year>.pdf`

## 2. Velocities

### 2.1 EKE Mean Map vs DUACS

This diagnostic plots the mean surface EKE map from DUACS and each model for the requested year.

EKE formula:

\[
EKE_i = \frac{1}{2}\overline{u_i'(t)^2+v_i'(t)^2}
\]

Model velocity is geostrophic velocity derived from SSH/zos. DUACS velocity and model-derived
velocity are masked for \(|lat| < 5^\circ\).

Figure:

- `Figures_observations/pdf_figures/maps/eke_mean_vs_duacs_<year>.pdf`

### 2.2 Global EKE Spatial Spectrum

This is present in the current evaluation. It is the first panel of the EKE spatial spectrum
figure.

Input field:

- DUACS and model EKE maps over the whole rollout common window.
- Each EKE map is a time-mean map computed from velocity anomalies.

Spectrum method:

1. Select a spatial region: Global, Gulf Stream, or Agulhas.
2. Fill missing values using the regional mean when needed.
3. Remove a fitted 2-D linear plane from the map.
4. Apply a 2-D Hann window.
5. Compute a 2-D FFT with forward normalization.
6. Convert to 2-D PSD:

\[
PSD_{2D} = |\hat{X}|^2 \, L_x L_y / C_{window}
\]

7. Exclude Fourier modes beyond the final retained bin edge, then bin by radial wavenumber and
   multiply by bin-center wavenumber to obtain the isotropic radial spectrum.
8. Report log10 curve RMSE versus DUACS on the overlapping positive wavenumber domain.

Metric key:

- `surface_eke_spatial_spectrum_log10_rmse_vs_duacs`, with `region` in `Global`, `Gulf Stream`, `Agulhas`

Figure:

- `Figures_observations/pdf_figures/spectrum/regional_eke_spatial_global_temporal_spectra_vs_duacs.pdf`

### 2.3 Global KE Temporal Spectrum

The temporal spectrum diagnostic is surface geostrophic KE, not the time spectrum of the
time-mean EKE map.

For each grid cell:

\[
KE_i(t) = \frac{1}{2}\left(u_i(t)^2+v_i(t)^2\right)
\]

For each region, the code builds a cos(latitude)-weighted regional mean KE time series:

\[
KE_R(t) =
\frac{\sum_{i \in R} \cos(lat_i) KE_i(t)}
{\sum_{i \in R} \cos(lat_i)}
\]

Then it computes a Welch temporal PSD:

- Sampling frequency: \(365.25 / \Delta t_{median}\), in cycles/year
- `scipy.signal.welch`
- linear detrending
- Hann window
- `nperseg = min(256, N)`
- `scaling = density`
- zero frequency is removed from the plotted/metric curve

For large global grids, the code regularly spatially subsamples before the regional mean if the
native selected grid has more than 100,000 points. Gulf Stream and Agulhas typically remain at
native resolution.

Metrics:

- `surface_geostrophic_ke_temporal_spectrum_log10_rmse_vs_duacs_raw_daily`
- `surface_geostrophic_ke_temporal_spectrum_log10_rmse_vs_duacs_5day_avg`

Both have `region` in `Global`, `Gulf Stream`, `Agulhas`.

Figure:

- `Figures_observations/pdf_figures/spectrum/ke_temporal_spectra_vs_duacs.pdf`

## 3. Temperature

### 3.1 SST Spatial / Isotropic Spectra

Both current SST spectrum figures use the same panel layout:
Global, Gulf Stream, Agulhas.

Input field:

- OISST surface temperature
- Model top-level `thetao`

Before spectrum calculation, the field is converted to SST anomaly relative to calendar-day
climatology:

\[
SSTA(t) = SST(t) - \overline{SST}_{calendar\ day}
\]

For each region:

1. Select the rectangular region.
2. Keep the largest fully finite ocean block when possible, otherwise fill residual gaps with
   wet-field means.
3. Compute per-time isotropic spectra using the same 2-D detrend + Hann + FFT + radial binning
   method described above.
4. Average spectra over time in batches.
5. Compute log10 curve RMSE versus OISST.

Metrics:

- `surface_ssta_spatial_spectrum_log10_rmse_vs_oisst_5day_avg`
- `surface_ssta_isotropic_spectrum_log10_rmse_vs_oisst_5day_avg`

Both have `region` in `Global`, `Gulf Stream`, `Agulhas`.

Figures:

- `Figures_observations/pdf_figures/spectrum/regional_sst_spatial_global_temporal_spectra_vs_oisst.pdf`
- `Figures_observations/pdf_figures/spectrum/temperature_isotropic_spectra_vs_oisst.pdf`

Implementation note:

- The current main run sets `oisst_daily = None`, so the SST isotropic spectrum figure uses
  5-day OISST only. Raw daily OISST is not included in the current main output.

### 3.2 OHC Total Time Series: Upper 700 m And 700-2000 m

Global OHC for a layer:

\[
OHC(t)=\rho_0 C_p \sum_{x,y,z \in layer} T(x,y,z,t)\Delta z A(x,y)
\]

Obs and model OHC totals are computed on their own native grids, because a global integral is
grid robust:

- ARGO-IAP uses inferred observation cell area and depth thickness from depth centers.
- Models use native `areacello`, `dz`, and wet mask.

The output is monthly averaged (`MS` resampling).

Metric:

\[
RMSE_t = \sqrt{\frac{1}{N}\sum_t (OHC_M(t)-OHC_O(t))^2}
\]

Metric key:

- `monthly_ohc_rmse`, with `depth` equal to `Surface (0-700m)` or `Intermediate (700-2000m)`

Figure:

- `Figures_observations/pdf_figures/timeseries/ohc_monthly_vs_argo_iap_<period>.pdf`

### 3.3 OHC Annual Maps / Bias Maps

Annual OHC per unit area maps are computed on the ARGO-IAP grid.

Per unit area:

\[
OHC_{area}(x,y) =
\rho_0 C_p \sum_{z \in layer} \overline{T}(x,y,z)\Delta z
\]

where \(\overline{T}\) is the annual mean for the requested year.

The map comparison figure plots ARGO-IAP and model annual OHC maps. The bias map figure plots:

\[
Bias(x,y)=OHC_{area,M}(x,y)-OHC_{area,O}(x,y)
\]

For map-comparison metrics, the same area-weighted map RMSE convention is used:

\[
RMSE_{map} = \sqrt{\langle Bias^2\rangle_A}
\]

Metric key:

- `annual_mean_ohc_per_area_map_rmse`, with layer/depth labels

Figures:

- `Figures_observations/pdf_figures/maps/ohc_layer_integrated_maps_vs_argo_iap_<year>.pdf`
- `Figures_observations/pdf_figures/maps/ohc_layer_integrated_bias_maps_vs_argo_iap_<year>.pdf`

## 4. Anomalies

The anomaly diagnostics are computed on time series for:

- global SST vs OISST
- global surface EKE vs DUACS
- global upper-700 m OHC vs ARGO-IAP

SST and EKE keep the 5-day cadence. Upper-700 m OHC is monthly.

### 4.1 Mean-Removed Anomaly Time Series

For each scalar time series:

\[
X'(t)=X(t)-\overline{X}
\]

Metric:

\[
RMSE = \sqrt{\frac{1}{N}\sum_t (X'_M(t)-X'_O(t))^2}
\]

Metric keys:

- `global_sst_mean_removed_anomaly_rmse`
- `global_surface_eke_mean_removed_anomaly_rmse`
- `global_ohc_upper700_mean_removed_anomaly_rmse`

Figures:

- `Figures_observations/pdf_figures/timeseries/global_sst_mean_removed_anomaly_vs_oisst.pdf`
- `Figures_observations/pdf_figures/timeseries/global_surface_eke_mean_removed_anomaly_vs_duacs.pdf`
- `Figures_observations/pdf_figures/timeseries/global_ohc_upper700_mean_removed_anomaly_vs_argo_iap_<period>.pdf`

### 4.2 Linear Trend Comparison

Each observation and model time series is fit independently with ordinary least squares:

\[
X(t)=a t + b + \epsilon(t)
\]

where \(t\) is years since the first timestamp, using `days / 365.25`.

The trend plot shows only the fitted OLS lines, centered at each line's first fitted value:

\[
\Delta \hat{X}(t)=\hat{X}(t)-\hat{X}(t_0)
\]

Metrics:

- model slope \(a_M\)
- model trend bias \(a_M-a_O\)

Metric keys:

- `<prefix>_linear_trend_per_year`
- `<prefix>_linear_trend_bias_per_year`

Prefixes:

- `global_sst`
- `global_surface_eke`
- `global_ohc_upper700`

Figures:

- `Figures_observations/pdf_figures/timeseries/global_sst_linear_trend_comparison_vs_oisst.pdf`
- `Figures_observations/pdf_figures/timeseries/global_surface_eke_linear_trend_comparison_vs_duacs.pdf`
- `Figures_observations/pdf_figures/timeseries/global_ohc_upper700_linear_trend_comparison_vs_argo_iap_<period>.pdf`

### 4.3 Detrended + Deseasonalized Residual Anomaly Variance

This is the variance metric of interest.

The residual anomaly calculation is:

1. Remove seasonal cycle.
2. Remove linear trend from the deseasonalized series.
3. Compute variance of the residual.

Seasonal-cycle rule:

- If median cadence is <= 10 days and enough repeated pentads exist, use a 365-day no-leap
  pentad climatology; dates after February are aligned across leap years and February 29 is
  assigned to the February 28 pentad.
- Otherwise, if all calendar months repeat, use monthly climatology.
- Otherwise, remove only the mean.

For a time series:

\[
X_{ds}(t)=X(t)-C_{season}(t)
\]

\[
r(t)=X_{ds}(t)-(\hat{a}t+\hat{b})
\]

\[
Var_{resid} = Var(r(t))
\]

The script uses `numpy.var`, so the variance is population variance (`ddof=0`).

Metrics:

- residual anomaly time-series RMSE:

\[
RMSE_r=\sqrt{\frac{1}{N}\sum_t (r_M(t)-r_O(t))^2}
\]

- residual variance:

\[
Var_M
\]

- residual variance bias:

\[
Var_M - Var_O
\]

- residual variance ratio:

\[
Var_M / Var_O
\]

Metric keys:

- `<prefix>_detrended_deseasonalized_anomaly_rmse`
- `<prefix>_detrended_deseasonalized_anomaly_variance`
- `<prefix>_detrended_deseasonalized_anomaly_variance_bias`
- `<prefix>_detrended_deseasonalized_anomaly_variance_ratio`

Figures:

- `Figures_observations/pdf_figures/timeseries/global_sst_detrended_deseasonalized_anomaly_vs_oisst.pdf`
- `Figures_observations/pdf_figures/timeseries/global_surface_eke_detrended_deseasonalized_anomaly_vs_duacs.pdf`
- `Figures_observations/pdf_figures/timeseries/global_ohc_upper700_detrended_deseasonalized_anomaly_vs_argo_iap_<period>.pdf`
- `Figures_observations/pdf_figures/variance/global_sst_residual_anomaly_variance_vs_oisst.pdf`
- `Figures_observations/pdf_figures/variance/global_surface_eke_residual_anomaly_variance_vs_duacs.pdf`
- `Figures_observations/pdf_figures/variance/global_ohc_upper700_residual_anomaly_variance_vs_argo_iap_<period>.pdf`

### 4.4 Maps Of Residual Anomaly Variance

These maps apply the same detrended + deseasonalized residual variance calculation independently
at every grid cell.

#### SST Residual Variance Map

Observation: OISST grid.

Model:

1. Compute residual variance on each model's native grid.
2. Interpolate the model variance map to the OISST grid.
3. Mask to finite OISST cells.

Map metric:

\[
RMSE_{varmap}=\sqrt{\langle (Var_M(x,y)-Var_O(x,y))^2\rangle_A}
\]

Area-weighted centered pattern correlation, using only cells where both fields and area are
finite and area is positive:

\[
Corr_A =
\frac{\sum_i A_i(Var_{M,i}-\overline{Var_M}_A)(Var_{O,i}-\overline{Var_O}_A)}
{\sqrt{\sum_i A_i(Var_{M,i}-\overline{Var_M}_A)^2
\sum_i A_i(Var_{O,i}-\overline{Var_O}_A)^2}}
\]

Metric keys:

- `surface_sst_residual_variance_map_rmse`
- `surface_sst_residual_variance_area_weighted_pattern_corr`

Figure:

- `Figures_observations/pdf_figures/variance/sst_residual_anomaly_variance_maps_vs_oisst.pdf`

#### Upper-700 m OHC Residual Variance Map

Observation and model fields are compared on the ARGO-IAP grid.

1. Compute OHC per unit area in the upper 700 m:

\[
OHC_{area}(x,y,t)=\rho_0 C_p \sum_{0}^{700m} T(x,y,z,t)\Delta z
\]

2. Monthly-average the OHC per-area field.
3. Remove seasonal cycle and trend at each grid cell.
4. Compute residual variance at each grid cell.
5. Compare model and observation variance maps with area-weighted RMSE and area-weighted
   centered pattern correlation.

Metric keys:

- `ohc_upper700_per_area_residual_variance_map_rmse`
- `ohc_upper700_per_area_residual_variance_area_weighted_pattern_corr`

Figure:

- `Figures_observations/pdf_figures/variance/ohc_upper700_residual_anomaly_variance_maps_vs_argo_iap_<period>.pdf`

## Main Figure/Metric Inventory

### Total RMSE + Spatial RMSE Maps

- Velocity vector RMSE map and total RMSE
- EKE RMSE map and total RMSE
- SST RMSE map and total RMSE
- OHC per-area RMSE maps and total RMSE for:
  - `Surface (0-700m)`
  - `Intermediate (700-2000m)`

### Velocities

- Mean EKE map vs DUACS
- EKE spatial spectra for:
  - Global
  - Gulf Stream
  - Agulhas
- Surface geostrophic KE temporal spectra for:
  - Global
  - Gulf Stream
  - Agulhas

### Temperature

- SST/SSTA spectra for:
  - Global
  - Gulf Stream
  - Agulhas
- Monthly total OHC time series for:
  - `Surface (0-700m)`
  - `Intermediate (700-2000m)`
- Annual OHC per-area maps and bias maps for:
  - `Surface (0-700m)`
  - `Intermediate (700-2000m)`

### Anomalies

- Mean-removed time-series anomalies for SST, EKE, and upper-700 m OHC
- OLS linear trend comparison for SST, EKE, and upper-700 m OHC
- Detrended + deseasonalized residual anomaly time series and residual variance for SST, EKE,
  and upper-700 m OHC
- Residual anomaly variance maps for SST and upper-700 m OHC

## Implementation Pointers

Useful functions in `obs_evaluation.py`:

- `compute_isotropic_spectrum_torch`: 2-D FFT and isotropic radial spectrum.
- `_compute_isotropic_spectrum_from_map`: spectrum from a 2-D map, used by EKE spatial spectra.
- `_compute_isotropic_spectrum_from_surface`: SSTA spectra from time-varying SST fields.
- `_compute_temporal_ke_spectrum_from_surface`: regional KE time series and Welch PSD.
- `_field_rmse_over_time`: per-grid-cell time RMSE.
- `_area_weighted_map_rmse`: scalar area-weighted map RMSE.
- `_surface_eke_from_uv`: time-mean EKE map.
- `_surface_eke_timeseries_from_uv`: area-weighted EKE time series.
- `_series_linear_trend_fit_per_year`: OLS trend fit and slope.
- `_series_detrended_deseasonalized_anomaly`: scalar residual anomaly.
- `_detrended_deseasonalized_dataarray`: grid-cell residual anomaly for maps.
- `_plot_sst_residual_variance_maps`: SST residual variance maps.
- `_plot_ohc_upper700_residual_variance_maps`: upper-700 m OHC residual variance maps.

The `main()` function calls these blocks in the same order used in this document.
