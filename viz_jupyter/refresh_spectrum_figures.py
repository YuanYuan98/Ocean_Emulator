"""Regenerate observation-evaluation spectrum figures only.

This is a lightweight entry point for updating
``Figures_observations/pdf_figures/spectrum`` after changing OISST or DUACS
preprocessed inputs, without recomputing maps, structure, and time-series
diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh observation spectrum figures")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--duacs-velocity-kind", choices=["absolute", "anomaly"], default="absolute")
    return parser.parse_args()


def _import_obs_evaluation():
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0]]
    try:
        import obs_evaluation as oe
    finally:
        sys.argv = old_argv
    return oe


def _prepared_obs_dataset(oe, key: str, year: int | None = None):
    ds = oe._infer_obs_grid_metrics(
        oe._drop_singleton_nonspatial_dims(
            oe._standardize_latlon_time(oe._maybe_open_dataset(oe.OBS_CONFIG[key]))
        )
    )
    if year is not None:
        ds = oe._year_slice(ds, year)
    return ds


def _prepared_obs_dataset_optional(oe, key: str, year: int | None = None):
    if not oe.OBS_CONFIG[key].exists():
        print(f"  WARNING: optional observation path not found: {oe.OBS_CONFIG[key]}")
        return None
    return _prepared_obs_dataset(oe, key, year)


def _load_emulators(oe):
    emulators = {}
    for path, label, color in oe.EMULATOR_CONFIGS:
        print(f" - {label}: {path}")
        emulators[label] = {"ds": oe._prepare_emulator_dataset(path), "color": color}
    oe._attach_common_model_wet_masks(emulators)
    return emulators


def _replace_spectrum_metrics(metrics_path: Path, spectrum_metrics: pd.DataFrame) -> None:
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        existing = existing[~existing["metric"].astype(str).str.contains("spectrum", na=False)]
        updated = pd.concat([existing, spectrum_metrics], ignore_index=True)
    else:
        updated = spectrum_metrics
    updated.to_csv(metrics_path, index=False)


def main() -> None:
    args = parse_args()
    oe = _import_obs_evaluation()

    print("Loading observations for spectrum diagnostics...")
    duacs = _prepared_obs_dataset(oe, "duacs", args.year)
    duacs_raw = _prepared_obs_dataset(oe, "duacs_raw", args.year)
    oisst = _prepared_obs_dataset(oe, "oisst", None)
    oisst_daily = _prepared_obs_dataset_optional(oe, "oisst_daily", None)

    print("Loading emulator rollouts...")
    emulators = _load_emulators(oe)

    print("Evaluating isotropic temperature spectra vs OISST...")
    temp_spec_metrics = oe._plot_temperature_spectra(oisst, emulators, oisst_daily=oisst_daily)
    oe.save_current_fig("temperature_isotropic_spectra_vs_oisst", oe.SPECTRUM_DIR)

    print("Evaluating temporal KE spectra vs raw and 5-day DUACS...")
    ke_spec_metrics = oe._plot_temporal_ke_spectra(
        duacs,
        duacs_raw,
        emulators,
        velocity_kind=args.duacs_velocity_kind,
    )
    oe.save_current_fig("ke_temporal_spectra_vs_duacs", oe.SPECTRUM_DIR)

    spectrum_metrics = pd.concat([temp_spec_metrics, ke_spec_metrics], ignore_index=True)
    spectrum_metrics_path = Path(oe.METRICS_DIR) / f"spectrum_metrics_{args.year}.csv"
    spectrum_metrics.to_csv(spectrum_metrics_path, index=False)
    print(f"  -> Saved spectrum metrics: {spectrum_metrics_path}")

    observation_metrics_path = Path(oe.METRICS_DIR) / f"observation_metrics_{args.year}.csv"
    _replace_spectrum_metrics(observation_metrics_path, spectrum_metrics)
    print(f"  -> Updated spectrum rows in: {observation_metrics_path}")


if __name__ == "__main__":
    main()
