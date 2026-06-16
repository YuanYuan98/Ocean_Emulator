"""Compatibility wrapper for downloading ARGO-IAP salinity files.

Prefer ``download_argo_iap.py`` for new temperature+salinity downloads.
"""

from __future__ import annotations

import sys

from download_argo_iap import main


if __name__ == "__main__":
    if "--year" in sys.argv:
        idx = sys.argv.index("--year")
        year = sys.argv[idx + 1]
        sys.argv[idx : idx + 2] = ["--start-year", year, "--end-year", year]
    has_field = any(arg == "--field" or arg.startswith("--field=") for arg in sys.argv)
    if not has_field:
        sys.argv.extend(["--field", "salinity"])
    main()
