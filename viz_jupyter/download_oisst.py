"""Download NOAA OISST v2.1 daily NetCDF files.

The observation evaluation currently keeps daily OISST as raw files under
``datasets/obs_data/oisst`` and derives 5-day aligned zarr products from those
files.  This downloader fills the raw daily archive and is intentionally
restartable: existing complete files are skipped by default.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


RAW_OBS_ROOT = Path("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/obs_data")
NOAA_OISST_BASE_URL = (
    "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/"
    "v2.1/access/avhrr"
)


@dataclass(frozen=True)
class DownloadTask:
    date: str
    url: str
    target: Path


@dataclass(frozen=True)
class DownloadResult:
    date: str
    path: str
    status: str
    bytes: int
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NOAA OISST v2.1 daily NetCDF files")
    parser.add_argument("--raw-obs-root", type=Path, default=RAW_OBS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-year", type=int, default=1982)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--base-url", default=NOAA_OISST_BASE_URL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-wait", type=float, default=5.0)
    parser.add_argument("--min-existing-bytes", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _iter_dates(start_year: int, end_year: int) -> list[str]:
    dates: list[str] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            for day in range(1, calendar.monthrange(year, month)[1] + 1):
                dates.append(f"{year:04d}{month:02d}{day:02d}")
    return dates


def _task_for_date(date: str, base_url: str, output_dir: Path) -> DownloadTask:
    year_month = date[:6]
    filename = f"oisst-avhrr-v02r01.{date}.nc"
    url = f"{base_url.rstrip('/')}/{year_month}/{filename}"
    return DownloadTask(date=date, url=url, target=output_dir / filename)


def _complete_existing(path: Path, min_existing_bytes: int) -> bool:
    return path.exists() and path.stat().st_size >= min_existing_bytes


def _download(task: DownloadTask, args: argparse.Namespace) -> DownloadResult:
    target = task.target
    if _complete_existing(target, args.min_existing_bytes) and not args.overwrite:
        return DownloadResult(task.date, str(target), "exists", target.stat().st_size)

    if args.dry_run:
        return DownloadResult(task.date, str(target), "dry-run", 0, task.url)

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    cmd = [
        "curl",
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--connect-timeout",
        str(min(args.timeout, 30.0)),
        "--max-time",
        str(args.timeout),
        "--retry",
        str(max(args.retries - 1, 0)),
        "--retry-delay",
        str(max(int(args.retry_wait), 0)),
        "--retry-all-errors",
        "--output",
        str(tmp),
        task.url,
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout * max(args.retries, 1) + args.retry_wait * max(args.retries - 1, 0) + 30,
        )
    except subprocess.TimeoutExpired as exc:
        if tmp.exists():
            tmp.unlink()
        return DownloadResult(task.date, str(target), "failed", 0, f"curl timeout: {exc}")

    if proc.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        return DownloadResult(task.date, str(target), "failed", 0, proc.stderr.strip())

    if tmp.stat().st_size < args.min_existing_bytes:
        size = tmp.stat().st_size
        tmp.unlink()
        return DownloadResult(task.date, str(target), "failed", 0, f"downloaded file is too small: {size} bytes")

    tmp.replace(target)
    return DownloadResult(task.date, str(target), "downloaded", target.stat().st_size)


def _write_manifest(output_dir: Path, results: list[DownloadResult], start_year: int, end_year: int) -> None:
    manifest = output_dir / f"manifest_oisst_{start_year}_{end_year}.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "path", "status", "bytes", "message"])
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.date):
            writer.writerow(
                {
                    "date": result.date,
                    "path": result.path,
                    "status": result.status,
                    "bytes": result.bytes,
                    "message": result.message,
                }
            )


def _write_readme(output_dir: Path, start_year: int, end_year: int, base_url: str) -> None:
    readme = output_dir / "README_oisst.md"
    readme.write_text(
        "NOAA OISST v2.1 daily NetCDF files for observation evaluation.\n\n"
        f"Source: {base_url.rstrip('/')}\n"
        f"Years requested: {start_year}-{end_year}\n\n"
        "Existing complete files are skipped by download_oisst.py unless --overwrite is used.\n"
        "The current obs_evaluation.py spectrum diagnostic reads the 5-day aligned product in "
        "nativegrid/oisst.zarr. Longer raw daily files here can be used to build SST/SSTA "
        "climatologies and longer OISST diagnostics.\n"
    )


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.retries < 1:
        raise ValueError("--retries must be >= 1")

    output_dir = args.output_dir or args.raw_obs_root / "oisst"
    dates = _iter_dates(args.start_year, args.end_year)
    tasks = [_task_for_date(date, args.base_url, output_dir) for date in dates]
    total = len(tasks)

    print(f"Downloading NOAA OISST v2.1 daily files for {args.start_year}-{args.end_year}")
    print(f"Output directory: {output_dir}")
    print(f"Tasks: {total}; workers: {args.workers}; dry_run={args.dry_run}")

    results: list[DownloadResult] = []
    completed = 0
    counts: dict[str, int] = {}
    bytes_written = 0
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_download, task, args) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            counts[result.status] = counts.get(result.status, 0) + 1
            if result.status in {"downloaded", "exists"}:
                bytes_written += result.bytes
            if result.status == "failed":
                print(f"  !! failed {result.date}: {result.message}")
            if completed == total or completed % 100 == 0:
                elapsed = max(time.time() - started, 1.0)
                rate = completed / elapsed
                print(
                    f"  -> {completed}/{total} complete "
                    f"({rate:.2f} files/s; counts={counts}; bytes={bytes_written / 1024**3:.2f} GiB)"
                )

    _write_manifest(output_dir, results, args.start_year, args.end_year)
    _write_readme(output_dir, args.start_year, args.end_year, args.base_url)

    failed = [result for result in results if result.status == "failed"]
    if failed:
        raise SystemExit(f"{len(failed)} OISST downloads failed; see manifest in {output_dir}")

    print("Download complete.")


if __name__ == "__main__":
    main()
