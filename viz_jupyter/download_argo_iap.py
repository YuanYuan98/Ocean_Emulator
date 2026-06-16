"""Download ARGO-IAP monthly temperature and salinity NetCDF files."""

from __future__ import annotations

import argparse
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


RAW_OBS_ROOT = Path("/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/obs_data")


@dataclass(frozen=True)
class FieldConfig:
    name: str
    base_urls: tuple[str, ...]
    filename_templates: tuple[str, ...]


@dataclass(frozen=True)
class DownloadTask:
    field: str
    year: int
    month: int
    base_urls: tuple[str, ...]
    filename_templates: tuple[str, ...]


FIELD_CONFIGS = {
    "temperature": FieldConfig(
        name="temperature",
        base_urls=(
            "http://www.ocean.iap.ac.cn/ftp/cheng/CZ16_v0_IAP_Temperature_0p5_gridded_1month_netcdf",
            "http://www.ocean.iap.ac.cn/ftp/cheng/CZ16_v0_IAP_temperature_0p5_gridded_1month_netcdf",
            "http://159.226.119.60/cheng/CZ16_v0_IAP_Temperature_0p5_gridded_1month_netcdf",
            "http://159.226.119.60/cheng/CZ16_v0_IAP_temperature_0p5_gridded_1month_netcdf",
        ),
        filename_templates=(
            "IAP_05_2000m_temperature_year_{year}_month_{month:02d}.nc",
            "IAP_05_2000m_Temperature_year_{year}_month_{month:02d}.nc",
            "IAP_05_2000m_temp_year_{year}_month_{month:02d}.nc",
        ),
    ),
    "salinity": FieldConfig(
        name="salinity",
        base_urls=(
            "http://www.ocean.iap.ac.cn/ftp/cheng/CZ16_v0_IAP_Salinity_0p5_gridded_1month_netcdf",
            "http://www.ocean.iap.ac.cn/ftp/cheng/CZ16_v0_IAP_salinity_0p5_gridded_1month_netcdf",
            "http://159.226.119.60/cheng/CZ16_v0_IAP_Salinity_0p5_gridded_1month_netcdf",
            "http://159.226.119.60/cheng/CZ16_v0_IAP_salinity_0p5_gridded_1month_netcdf",
        ),
        filename_templates=(
            "IAP_05_2000m_salinity_year_{year}_month_{month:02d}.nc",
            "IAP_05_2000m_Salinity_year_{year}_month_{month:02d}.nc",
            "IAP_05_2000m_salt_year_{year}_month_{month:02d}.nc",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ARGO-IAP monthly NetCDF files")
    parser.add_argument("--raw-obs-root", type=Path, default=RAW_OBS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument(
        "--field",
        action="append",
        choices=sorted(FIELD_CONFIGS),
        default=[],
        help="Field to download. Can be repeated. Defaults to both temperature and salinity.",
    )
    parser.add_argument(
        "--base-url",
        action="append",
        default=[],
        help="Override base URL. Applies to all requested fields; mostly useful for debugging.",
    )
    parser.add_argument(
        "--filename-template",
        action="append",
        default=[],
        help="Override filename template. Applies to all requested fields; mostly useful for debugging.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=5.0)
    parser.add_argument("--min-existing-bytes", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent downloads")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _is_complete_existing(path: Path, min_existing_bytes: int) -> bool:
    return path.exists() and path.stat().st_size >= min_existing_bytes


def _download_url(
    url: str,
    target: Path,
    *,
    overwrite: bool,
    timeout: float,
    retries: int,
    retry_wait: float,
    min_existing_bytes: int,
    dry_run: bool,
) -> bool:
    if _is_complete_existing(target, min_existing_bytes) and not overwrite:
        print(f"  -> exists: {target.name}")
        return True

    if dry_run:
        print(f"  -> would try {url}")
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    for attempt in range(1, retries + 1):
        try:
            print(f"  -> trying {url} (attempt {attempt}/{retries})")
            with urllib.request.urlopen(url, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if status >= 400:
                    return False
                with tmp.open("wb") as out:
                    shutil.copyfileobj(response, out)
            if tmp.stat().st_size < min_existing_bytes:
                raise OSError(f"downloaded file is too small: {tmp.stat().st_size} bytes")
            tmp.replace(target)
            print(f"  -> wrote {target} ({target.stat().st_size / 1024**2:.1f} MiB)")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            if tmp.exists():
                tmp.unlink()
            print(f"     failed: {exc}")
            if attempt < retries:
                time.sleep(retry_wait)
    return False


def _field_urls(args: argparse.Namespace, config: FieldConfig) -> tuple[list[str], list[str]]:
    base_urls = args.base_url or list(config.base_urls)
    filename_templates = args.filename_template or list(config.filename_templates)
    return base_urls, filename_templates


def _download_task(args: argparse.Namespace, output_dir: Path, task: DownloadTask) -> None:
    for template in task.filename_templates:
        filename = template.format(year=task.year, month=task.month)
        target = output_dir / filename
        for base_url in task.base_urls:
            url = f"{base_url.rstrip('/')}/{filename}"
            if _download_url(
                url,
                target,
                overwrite=args.overwrite,
                timeout=args.timeout,
                retries=args.retries,
                retry_wait=args.retry_wait,
                min_existing_bytes=args.min_existing_bytes,
                dry_run=args.dry_run,
            ):
                return
    raise RuntimeError(f"Could not download {task.field} file for {task.year}-{task.month:02d}")


def _write_readme(output_dir: Path, fields: list[str], start_year: int, end_year: int) -> None:
    readme = output_dir / "README_argo_iap.md"
    readme.write_text(
        "ARGO-IAP monthly NetCDF files downloaded for obs_evaluation.py.\n\n"
        f"Fields: {', '.join(fields)}\n"
        f"Years: {start_year}-{end_year}\n\n"
        "Run prepare_obs_5day_aligned.py --products argo-iap after downloading to rebuild "
        "nativegrid/argo-iap.zarr.\n"
    )


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")
    if args.retries < 1:
        raise ValueError("--retries must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    output_dir = args.output_dir or args.raw_obs_root / "argo-iap"
    fields = list(dict.fromkeys(args.field or ["temperature", "salinity"]))

    print(f"Downloading ARGO-IAP fields {fields} for {args.start_year}-{args.end_year}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.workers}")
    tasks: list[DownloadTask] = []
    for field in fields:
        config = FIELD_CONFIGS[field]
        base_urls, filename_templates = _field_urls(args, config)
        for year in range(args.start_year, args.end_year + 1):
            for month in range(1, 13):
                tasks.append(
                    DownloadTask(
                        field=field,
                        year=year,
                        month=month,
                        base_urls=tuple(base_urls),
                        filename_templates=tuple(filename_templates),
                    )
                )

    total = len(tasks)
    if args.workers == 1:
        for completed, task in enumerate(tasks, start=1):
            _download_task(args, output_dir, task)
            print(f"Completed {completed}/{total}: {task.field} {task.year}-{task.month:02d}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(_download_task, args, output_dir, task): task for task in tasks}
            for completed, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception:
                    for pending in future_to_task:
                        pending.cancel()
                    raise
                print(f"Completed {completed}/{total}: {task.field} {task.year}-{task.month:02d}")

    if not args.dry_run:
        _write_readme(output_dir, fields, args.start_year, args.end_year)
    print("Done.")


if __name__ == "__main__":
    main()
