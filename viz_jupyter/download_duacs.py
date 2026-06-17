"""Download raw daily DUACS for observation temporal spectra.

The Copernicus Marine toolbox requires a free Copernicus Marine account. Set
credentials with either:

    export COPERNICUSMARINE_SERVICE_USERNAME=...
    export COPERNICUSMARINE_SERVICE_PASSWORD=...

or provide a toolbox-compatible credentials file with ``--credentials-file``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path


RAW_OBS_ROOT = Path("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/obs_data")
DUACS_DATASET_ID = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"
DUACS_VELOCITY_VARIABLES = ["ugos", "vgos", "ugosa", "vgosa"]
DUACS_SSH_VARIABLES = ["adt", "sla"]
DEFAULT_START = "2014-10-18"
DEFAULT_END = "2022-12-26"
DEFAULT_FILENAME = (
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_multi-vars_"
    "179.94W-179.94E_89.94S-89.94N_2014-10-18-2022-12-26.zarr"
)
DEFAULT_COLLECTION = (
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_velocity_"
    "2014-10-18-2022-12-26_yearly"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw daily DUACS zarr for obs_evaluation.py")
    parser.add_argument("--raw-obs-root", type=Path, default=RAW_OBS_ROOT)
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--output-filename", default=DEFAULT_FILENAME)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION)
    parser.add_argument("--credentials-file", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output zarr.")
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Download one large zarr instead of yearly zarrs. Yearly mode is the default.",
    )
    parser.add_argument(
        "--include-ssh",
        action="store_true",
        help="Also download adt/sla. Temporal KE spectra only need velocity variables.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ask Copernicus Marine to validate the request without downloading data.",
    )
    return parser.parse_args()


def _has_credentials(args: argparse.Namespace) -> bool:
    if args.credentials_file is not None and args.credentials_file.exists():
        return True
    if os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME") and os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD"):
        return True
    home = Path.home()
    return any(
        path.exists()
        for path in [
            home / ".copernicusmarine" / ".copernicusmarine-credentials",
            home / ".netrc",
            home / "_netrc",
        ]
    )


def _yearly_ranges(start: str, end: str) -> list[tuple[date, date]]:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    if start_date > end_date:
        raise ValueError(f"Invalid date range: {start} > {end}")
    ranges = []
    current = start_date
    while current <= end_date:
        year_end = min(date(current.year, 12, 31), end_date)
        ranges.append((current, year_end))
        current = year_end + timedelta(days=1)
    return ranges


def _subset_command(
    args: argparse.Namespace,
    out_dir: Path,
    output_filename: str,
    start_date: str,
    end_date: str,
    variables: list[str],
) -> list[str]:
    cmd = [
        "uvx",
        "--with",
        "h5py",
        "copernicusmarine",
        "subset",
        "-i",
        DUACS_DATASET_ID,
        "--start-datetime",
        start_date,
        "--end-datetime",
        end_date,
        "--file-format",
        "zarr",
        "--output-directory",
        str(out_dir),
        "--output-filename",
        output_filename,
        "--disable-progress-bar",
    ]
    for variable in variables:
        cmd.extend(["-v", variable])
    if args.credentials_file is not None:
        cmd.extend(["--credentials-file", str(args.credentials_file)])
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _run_command(cmd: list[str]) -> None:
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _download_single(args: argparse.Namespace, variables: list[str]) -> None:
    out_dir = args.raw_obs_root / "duacs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_filename

    if out_path.exists() and not args.overwrite and not args.dry_run:
        print(f"Output already exists, skipping: {out_path}")
        return

    cmd = _subset_command(args, out_dir, args.output_filename, args.start_date, args.end_date, variables)
    if args.overwrite:
        cmd.append("--overwrite")
    _run_command(cmd)
    if not args.dry_run:
        print(f"Downloaded DUACS zarr: {out_path}")


def _download_yearly(args: argparse.Namespace, variables: list[str]) -> None:
    collection_dir = args.raw_obs_root / "duacs" / args.collection_name
    collection_dir.mkdir(parents=True, exist_ok=True)
    ranges = _yearly_ranges(args.start_date, args.end_date)
    for start_date, end_date in ranges:
        final_name = f"duacs_{start_date:%Y%m%d}_{end_date:%Y%m%d}.zarr"
        temp_name = f"duacs_{start_date:%Y%m%d}_{end_date:%Y%m%d}.tmp.zarr"
        final_path = collection_dir / final_name
        temp_path = collection_dir / temp_name
        if final_path.exists() and not args.overwrite and not args.dry_run:
            print(f"Output already exists, skipping: {final_path}")
            continue
        if not args.dry_run and temp_path.exists():
            shutil.rmtree(temp_path)
        if not args.dry_run and args.overwrite and final_path.exists():
            shutil.rmtree(final_path)

        output_name = final_name if args.dry_run else temp_name
        cmd = _subset_command(
            args,
            collection_dir,
            output_name,
            start_date.isoformat(),
            end_date.isoformat(),
            variables,
        )
        if not args.dry_run:
            cmd.append("--overwrite")
        _run_command(cmd)
        if not args.dry_run:
            if not temp_path.exists():
                raise FileNotFoundError(f"Expected Copernicus output was not written: {temp_path}")
            temp_path.rename(final_path)
            print(f"Downloaded DUACS yearly zarr: {final_path}")
    if not args.dry_run:
        print(f"Downloaded DUACS yearly collection: {collection_dir}")


def main() -> None:
    args = parse_args()
    if not _has_credentials(args):
        raise SystemExit(
            "Copernicus Marine credentials are required. Set "
            "COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD, "
            "or pass --credentials-file."
        )
    variables = DUACS_VELOCITY_VARIABLES + (DUACS_SSH_VARIABLES if args.include_ssh else [])
    if args.single_file:
        _download_single(args, variables)
    else:
        _download_yearly(args, variables)


if __name__ == "__main__":
    main()
