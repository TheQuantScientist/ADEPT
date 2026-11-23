#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${REVIEWER2_PYTHON:-/home/nckh2/qa/.venv/bin/python}"
INFERENCE_ROOT="${REVIEWER2_INFERENCE_ROOT:-${PROJECT_ROOT}/revision/reviewer2/parallel_20261122_083519}"
OUTPUT_ROOT="${REVIEWER2_OUTPUT:-${PROJECT_ROOT}/revision/reviewer2/three_baselines_results}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
printf 'Aggregating the 12 completed ADEPT checkpoint inference runs...\n'
"${PYTHON_BIN}" -u revision/reviewer2/run_reviewer2.py --mode aggregate --output "${INFERENCE_ROOT}"
printf 'Computing paired confidence intervals for APN, GraFITi, and tPatchGNN...\n'
"${PYTHON_BIN}" -u revision/reviewer2/run_three_baselines.py --output "${OUTPUT_ROOT}" --bootstrap-reps 5000
printf 'Finished. Paired analyses: %s\nADEPT inference summaries: %s\n' "${OUTPUT_ROOT}" "${INFERENCE_ROOT}"
