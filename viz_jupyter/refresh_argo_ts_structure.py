from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.argv = ["obs_evaluation.py"]
import obs_evaluation as oe  # noqa: E402


def main() -> None:
    print("Loading ARGO and emulator datasets...", flush=True)
    argo_iap = oe._drop_singleton_nonspatial_dims(
        oe._standardize_latlon_time(oe._maybe_open_dataset(oe.OBS_CONFIG["argo_iap"]))
    )
    if {"lat", "lon"}.issubset(argo_iap.dims) and "area" not in argo_iap:
        argo_iap = oe._infer_obs_grid_metrics(argo_iap)

    emulators = {}
    for path, label, color in oe.EMULATOR_CONFIGS:
        print(f" - {label}", flush=True)
        emulators[label] = {"ds": oe._prepare_emulator_dataset(path), "color": color}
    oe._attach_common_model_wet_masks(emulators)

    ohc_start, ohc_end = oe._resolve_ohc_period(
        argo_iap,
        emulators,
        oe.args.ohc_start_date,
        oe.args.ohc_end_date,
    )
    ohc_label = oe._period_label(ohc_start, ohc_end)
    ohc_slug = oe._period_slug(ohc_start, ohc_end)
    print(f"Using period {ohc_label}", flush=True)

    map_metrics = []
    for spec in oe.ARGO_STRUCTURE_VARIABLES:
        print(f"Refreshing {spec['title']} maps...", flush=True)
        map_metrics.append(oe._plot_argo_variable_layer_maps(argo_iap, emulators, oe.args.year, spec))
        oe.save_current_fig(f"{spec['filename']}_layer_mean_maps_vs_argo_iap_{oe.args.year}", oe.MAP_DIR)

    print("Refreshing OHC maps...", flush=True)
    map_metrics.append(oe._plot_argo_ohc_layer_maps(argo_iap, emulators, oe.args.year))
    oe.save_current_fig(f"ohc_layer_integrated_maps_vs_argo_iap_{oe.args.year}", oe.MAP_DIR)

    layer_metrics = []
    structure_metrics = []
    for spec in oe.ARGO_STRUCTURE_VARIABLES:
        print(f"Refreshing {spec['title']} time series...", flush=True)
        obs_layer, sim_layer = oe._collect_layer_mean_timeseries(
            argo_iap,
            emulators,
            spec,
            ohc_start,
            ohc_end,
        )
        layer_metrics.append(oe._plot_layer_mean_timeseries(obs_layer, sim_layer, emulators, spec, ohc_label))
        oe.save_current_fig(f"{spec['filename']}_monthly_vs_argo_iap_{ohc_slug}", oe.TIMESERIES_DIR)

        layer_metrics.append(oe._plot_layer_mean_anomaly_timeseries(obs_layer, sim_layer, emulators, spec, ohc_label))
        oe.save_current_fig(f"{spec['filename']}_anomaly_vs_argo_iap_{ohc_slug}", oe.TIMESERIES_DIR)

        layer_metrics.append(oe._plot_layer_mean_monthly_climatology(obs_layer, sim_layer, emulators, spec, ohc_label))
        oe.save_current_fig(f"{spec['filename']}_monthly_climatology_vs_argo_iap_{ohc_slug}", oe.TIMESERIES_DIR)

        layer_metrics.append(oe._plot_layer_mean_deseasoned_anomaly_timeseries(obs_layer, sim_layer, emulators, spec, ohc_label))
        oe.save_current_fig(f"{spec['filename']}_deseasoned_anomaly_vs_argo_iap_{ohc_slug}", oe.TIMESERIES_DIR)

        print(f"Refreshing {spec['title']} zonal mean...", flush=True)
        structure_metrics.append(oe._plot_argo_variable_zonal_mean(argo_iap, emulators, oe.args.year, spec))
        oe.save_current_fig(f"argo_{spec['filename']}_zonal_mean_{oe.args.year}", oe.STRUCTURE_DIR)

        print(f"Refreshing {spec['title']} structure...", flush=True)
        structure_metrics.append(oe._plot_argo_variable_zonal_mean_bias(argo_iap, emulators, oe.args.year, spec))
        oe.save_current_fig(f"argo_{spec['filename']}_zonal_mean_bias_{oe.args.year}", oe.STRUCTURE_DIR)

    print("Refreshing OHC zonal mean...", flush=True)
    structure_metrics.append(oe._plot_argo_ohc_zonal_mean(argo_iap, emulators, oe.args.year))
    oe.save_current_fig(f"argo_ohc_zonal_mean_{oe.args.year}", oe.STRUCTURE_DIR)

    print("Refreshing OHC structure...", flush=True)
    ohc_structure_metrics = oe._plot_argo_ohc_zonal_mean_bias(argo_iap, emulators, oe.args.year)
    oe.save_current_fig(f"argo_ohc_zonal_mean_bias_{oe.args.year}", oe.STRUCTURE_DIR)

    old_combined = Path(oe.STRUCTURE_DIR) / f"argo_ts_zonal_mean_bias_{oe.args.year}.pdf"
    if old_combined.exists():
        old_combined.unlink()
        print(f"Removed old combined structure figure: {old_combined}", flush=True)

    new_metrics = pd.concat([*map_metrics, *layer_metrics, *structure_metrics, ohc_structure_metrics], ignore_index=True)
    metrics_path = Path(oe.METRICS_DIR) / f"observation_metrics_{oe.args.year}.csv"
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        replace_prefixes = (
            "monthly_temperature_layer_mean",
            "monthly_salinity_layer_mean",
            "annual_mean_temperature",
            "annual_mean_salinity",
            "annual_mean_ohc_per_area",
        )
        keep = ~existing["metric"].astype(str).apply(
            lambda value: any(value.startswith(prefix) for prefix in replace_prefixes)
        )
        metrics = pd.concat([existing[keep], new_metrics], ignore_index=True)
    else:
        metrics = new_metrics
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved merged metrics: {metrics_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
