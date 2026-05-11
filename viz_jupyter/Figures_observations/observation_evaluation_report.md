# Observation Evaluation Report

This report summarizes the figures and metrics in `Figures_observations/pdf_figures`.
The evaluation compares the Samudra 2 1/4 degree rollout and the OM4 1/4 degree
simulation against observation-based products for 2022.

## Compared Models

- Samudra 2 (1/4 degree): emulator rollout from `predictions.zarr`
- OM4 (1/4 degree): reference ocean simulation from `OM4.zarr`

## Observation Datasets

- DUACS: satellite-altimetry product used for surface geostrophic velocity and
  geostrophic eddy kinetic energy (EKE).
- OISST: daily observed sea-surface temperature product used for regional
  surface-temperature spatial spectra.
- ARGO-IAP: monthly ocean heat-content / temperature product used for global
  monthly OHC comparisons down to 2000 m.

## Preprocessing

DUACS and OISST are daily products. They were converted to the model's 5-day
rollout cadence by applying a centered 5-sample rolling mean, sampling on the
Samudra rollout dates, and relabeling to the exact rollout timestamps. The
resulting comparison products have 72 timestamps for 2022.

ARGO-IAP is monthly and was kept monthly. It is not forced onto the 5-day model
timeline, because that would add artificial high-frequency information.

No spatial regridding is applied. Each observation product is kept on its
source native grid (DUACS ~0.125°, OISST ~0.25°, ARGO-IAP ~0.5°) and the
diagnostics are computed on each side's own grid. Spectra use each grid's own
`dx`/`dy`; OHC uses each grid's own cell area and layer thickness.

## Evaluations Performed

### Surface geostrophic EKE vs DUACS

Files:

- `maps/eke_mean_vs_duacs_2022.pdf`
- `maps/eke_mean_lowpass_vs_duacs_2022.pdf`

DUACS geostrophic velocities are compared with model geostrophic velocities
derived from sea-surface height (`zos`). EKE is computed as
`0.5 * (u_g^2 + v_g^2)`. The equatorial band `|lat| <= 5 degrees` is masked.
The report includes both raw time-mean EKE and a Gaussian low-pass-filtered
version using `sigma = 0.5 degree`.

### Surface temperature isotropic spectra vs OISST

File:

- `spectrum/temperature_isotropic_spectra_vs_oisst.pdf`

Surface temperature anomalies are compared in four energetic regions: Gulf
Stream, Kuroshio, Agulhas, and Malvinas. The spectra are computed from regional
surface temperature anomalies after removing the time mean, applying linear
detrending and a Hann window. Spectral RMSE is computed on the spectral values
after interpolation to a common wavenumber axis.

### Surface geostrophic KE temporal spectra vs DUACS

File:

- `spectrum/ke_temporal_spectra_vs_duacs.pdf`

Regional temporal spectra are computed from surface geostrophic KE in the same
four regions. The model geostrophic velocities are derived from `zos`, while
DUACS provides observed geostrophic velocities. Welch spectra are computed along
time, and RMSE is calculated after interpolation to the DUACS frequency axis.

### Monthly global OHC vs ARGO-IAP

Files:

- `timeseries/ohc_monthly_vs_argo_iap_2022.pdf`
- `timeseries/ohc_anomaly_vs_argo_iap_2022.pdf`

Global monthly OHC is compared in two depth layers:

- Surface: 0-700 m
- Intermediate: 700-2000 m

The 2000-7000 m layer is excluded because ARGO-IAP only extends to 2000 m in
this setup. OHC is computed from temperature, cell area, layer thickness,
seawater density, and heat capacity. Both absolute monthly OHC RMSE and monthly
OHC anomaly RMSE are reported.

## Main Results

Both Samudra 2 and OM4 substantially underestimate surface geostrophic EKE
relative to DUACS. DUACS global mean EKE is `1.52e-02 m2/s2`; Samudra 2 is
`3.77e-03`, and OM4 is `4.02e-03`. After low-pass filtering, DUACS is
`5.61e-03`, Samudra 2 is `1.14e-03`, and OM4 is `1.71e-03`. OM4 is closer to
DUACS in mean EKE, especially in the low-pass field.

For surface temperature isotropic spectra, OM4 has lower RMSE than Samudra 2 in
all four regions. The largest Samudra 2 disadvantage is in the Gulf Stream; the
Kuroshio difference is small.

For surface geostrophic KE temporal spectra, both models are much lower than
DUACS. OM4 is slightly better than Samudra 2 in all four regions, but the
difference is small, roughly 1.6-2.7 percent in RMSE.

For absolute monthly OHC, OM4 is much closer in the surface 0-700 m layer, while
Samudra 2 is slightly closer in the intermediate 700-2000 m layer. Both models
are biased high relative to ARGO-IAP in the intermediate layer.

For monthly OHC anomalies, Samudra 2 is better in both layers. Surface anomaly
RMSE is `8.41e21 J` for Samudra 2 and `1.29e22 J` for OM4. Intermediate anomaly
RMSE is `4.57e21 J` for Samudra 2 and `4.68e21 J` for OM4.

## Conclusion

The observation comparisons show a split result. OM4 is generally closer to the
observation products for absolute surface dynamical statistics, especially EKE
and spectral comparisons. Samudra 2 is more competitive for heat-content
variability and performs better for OHC anomalies. The dominant shared weakness
is that both Samudra 2 and OM4 are under-energetic relative to DUACS, indicating
that surface mesoscale variability is too weak in both products.

