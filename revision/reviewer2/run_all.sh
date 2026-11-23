#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${REVIEWER2_PYTHON:-/home/nckh2/qa/.venv/bin/python}"
OUTPUT_ROOT="${REVIEWER2_OUTPUT:-${PROJECT_ROOT}/revision/reviewer2/results}"
mkdir -p "${OUTPUT_ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

printf 'Step 1/4: paired analysis of existing baseline exports\n'
"${PYTHON_BIN}" -u "${PROJECT_ROOT}/revision/reviewer2/run_reviewer2.py" --mode paired --output "${OUTPUT_ROOT}"

printf 'Step 2/4: original and perturbed inference with saved ADEPT control checkpoints\n'
"${PYTHON_BIN}" -u "${PROJECT_ROOT}/revision/reviewer2/run_reviewer2.py" --mode inference --device cuda --output "${OUTPUT_ROOT}"

printf 'Step 3/4: train three NEW MIMIC Warpformer runs and export predictions\n'
printf 'These are new runs and are not the original Table 1 checkpoints.\n'
cd "${PROJECT_ROOT}/vendor/upstream_benchmark"
"${PYTHON_BIN}" -u main.py \
  --gpu_id 0 --is_training 1 --loss MSE \
  --model_name Warpformer --model_id reviewer2_warpformer_mimic \
  --dataset_name MIMIC_III --dataset_root_path storage/datasets/MIMIC_III \
  --features M --seq_len 72 --pred_len 3 --enc_in 96 --c_out 96 \
  --d_model 256 --n_heads 4 --dropout 0 --n_layers 2 --use_multi_gpu 0 \
  --train_epochs 300 --patience 5 --val_interval 1 --itr 3 \
  --batch_size 16 --learning_rate 0.001 --num_workers 0 \
  --save_prediction_arrays 1 --checkpoints "${OUTPUT_ROOT}/warpformer"

printf 'Step 4/4: recompute paired analysis, including new MIMIC Warpformer predictions\n'
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -u "${PROJECT_ROOT}/revision/reviewer2/run_reviewer2.py" --mode paired --output "${OUTPUT_ROOT}"
printf 'Finished. Results: %s\n' "${OUTPUT_ROOT}"
