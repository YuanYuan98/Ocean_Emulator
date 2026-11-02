#!/usr/bin/env bash
set -euo pipefail
repo="${1:?repository}"
root="${2:?comparison root}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$root/mpl_cache"
export PATH="$repo/.venv/bin:$PATH"
cd "$repo"
"$repo/.venv/bin/python" -u scripts/supplement_revision_paper.py --root "$root" --render-only
"$repo/.venv/bin/python" -u scripts/verify_revision_supplement.py "$root"
