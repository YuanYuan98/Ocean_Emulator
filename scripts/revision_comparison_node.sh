#!/usr/bin/env bash
set -euo pipefail
output="${1:?output directory}"
python_bin="${2:?python interpreter}"
shift 2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export MPLBACKEND=Agg
export PATH="$(dirname "$python_bin"):$PATH"
export MPLCONFIGDIR="$output/mpl_cache"
cd "$output"
exec > "$output/diagnostics_${SLURM_JOB_ID}.log" 2>&1
"$python_bin" -u "$output/runtime/compare_revision_baselines.py" --output "$output" --models "$@"
"$python_bin" - "$output/comparison.ipynb" <<'PY'
import sys
import nbformat
from nbclient import NotebookClient
path = sys.argv[1]
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=600, kernel_name='python3').execute()
nbformat.write(notebook, path)
print('Executed notebook:', path, flush=True)
PY
