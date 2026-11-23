<div align="center">

# ADEPT

### Query-conditioned forecasting for irregular multivariate time series

Represent asynchronous observations as events, retrieve evidence for each
forecasting query, and combine neural predictions with adaptive historical experts.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-Irregular%20Time%20Series-6C63FF)
![Datasets](https://img.shields.io/badge/Benchmarks-4-2EA44F)

[Overview](#overview) · [Getting started](#getting-started) ·
[Experiments](#experiments) · [Diagnostics](#diagnostics) ·
[Repository](#repository-structure)

</div>

---

## Overview

ADEPT forecasts values at requested future times and variables without first
converting irregular observations into a regular grid. Its decoder combines a
neural prediction with history-derived scalar experts using query-dependent
mixture weights and a residual correction.

```text
  Irregular observations                   Future time–variable query
            │                                          │
            ▼                                          │
    Event representation                               │
            │                                          │
            ▼                                          ▼
  Transformer event memory ──► Variable-aware, time-sensitive retrieval
                                                        │
                       Historical experts ──────────────┤
                                                        ▼
                                      Adaptive expert decoder
                                                        │
                                                        ▼
                                                   Forecast
```

| Component | Role |
|:--|:--|
| **Event memory** | Encodes observed time–variable–value events with a transformer. |
| **Variable-aware retrieval** | Uses learned variable relations and continuous-time delta kernels to retrieve query-specific evidence. |
| **Historical experts** | Provide latest-value, exponential-memory, trend, graph-aware, and global-memory estimates. |
| **Adaptive decoder** | Combines the neural and historical estimates for each query. |

> **Naming:** The repository is called `Chronoa` for continuity with earlier
> experiments. The model and Python package are named **ADEPT**; some scripts
> and output filenames retain the earlier `chronoa` prefix.

## Getting started

### 1. Install

Python 3.10 or newer is required. From a clone of this repository:

```bash
cd Chronoa
python -m venv .venv
source .venv/bin/activate
pip install -e ".[benchmark]"
```

The benchmark extra installs PyTorch 2.6.0 and the dependencies declared in
[`pyproject.toml`](pyproject.toml). A CUDA-capable setup is useful for full
experiments; the runner also accepts `--device cpu`.

### 2. Prepare the benchmark data

| Runner name | Dataset | Preparation |
|:--|:--|:--|
| `P12` | PhysioNet Challenge 2012 | Public data; the upstream loader supports download and caching. |
| `MIMIC` | MIMIC-III, De Brouwer protocol | Credentialed access and preprocessing are required. |
| `USHCN` | US climate observations | Public data; the upstream loader supports download and caching. |
| `HumanActivity` | Wearable-sensor activity data | Public data; the upstream loader supports download and preprocessing. |

> [!IMPORTANT]
> The benchmark loader directory `vendor/upstream_benchmark/data/` is ignored
> by Git and is **not included in a fresh clone**. Before running an experiment,
> supply a compatible copy of the benchmark `data/` package at that path; the
> corresponding upstream implementation is available in
> [APN](https://github.com/decisionintelligence/APN). Data-loader versions and
> preprocessing must match the intended experiment. Dataset files and cached
> results are also excluded from this repository.

The vendored [benchmark setup instructions](vendor/upstream_benchmark/README.md)
describe dataset preparation, including the MIMIC preprocessing and file
placement steps. Follow the dataset providers' access conditions.

### 3. Run a forecast

After the benchmark loaders and data are available, train and evaluate ADEPT
on P12:

```bash
python scripts/experiments/run_chronoa_forecasting.py \
  --dataset P12 \
  --architecture-ablation apex \
  --loss-mode default \
  --random-seed 2024 \
  --output-root results/adept_quickstart \
  --device cuda \
  --epochs 100 \
  --patience 16 \
  --batch-size 8 \
  --learning-rate 0.0006 \
  --dropout 0.08 \
  --delta-kernels 6 \
  --max-events 2048 \
  --max-queries-per-sample 768 \
  --overwrite
```

Use `--device cpu` on a CPU-only machine. For a smaller smoke test, add
`--max-train-samples`, `--max-validation-samples`, and `--max-test-samples`;
results from a smoke test are not benchmark results.

## Experiments

<details>
<summary><b>Run all four datasets</b></summary>

```bash
python scripts/experiments/run_chronoa_forecasting.py \
  --all \
  --architecture-ablation apex \
  --output-root results/adept_standard \
  --device cuda \
  --overwrite
```

This runs the configured benchmark splits; prepare MIMIC and the other datasets
before launching the full set.

</details>

<details>
<summary><b>Common runner options</b></summary>

| Option | Purpose |
|:--|:--|
| `--dataset` / `--all` | Select one or more datasets, or all four. |
| `--architecture-ablation` | Select `apex` for ADEPT or an available architecture control. |
| `--loss-mode` | Select the default objective or the `mse` control. |
| `--history-perturbation` | Evaluate a configured history perturbation. |
| `--random-seed` | Set the experiment seed. |
| `--save-gate-log` | Export per-query gate weights and expert values. |
| `--device` | Use `auto`, `cpu`, `cuda`, or a PyTorch device string. |
| `--output-root` | Choose where to write experiment artifacts. |

For all options and supported choices, run:

```bash
python scripts/experiments/run_chronoa_forecasting.py --help
```

</details>

Revision experiment runners and their requirements are described in
[`revision/README.md`](revision/README.md) and
[`revision/reviewer2/README.md`](revision/reviewer2/README.md). Some analyses
require separately saved checkpoints, prediction arrays, or local paths and
cannot be reproduced from the repository alone.

## Diagnostics

Run a model with gate logging enabled, then summarize the saved gate behavior:

```bash
python scripts/experiments/run_chronoa_forecasting.py \
  --dataset MIMIC \
  --architecture-ablation apex \
  --save-gate-log \
  --output-root results/adept_gate_logs \
  --device cuda \
  --overwrite

python scripts/analysis/adept_gate_analysis.py \
  --root results/adept_gate_logs \
  --output-dir results/adept_gate_logs/gate_summaries
```

Gate weights describe how the model routes predictions; by themselves they do
not establish that an expert causes an accuracy improvement.

<details>
<summary><b>Objective-control and gate-log runs</b></summary>

```bash
CHRONOA_ROOT="$PWD" \
RUNNER="scripts/experiments/run_chronoa_forecasting.py" \
GATE_ANALYSIS="scripts/analysis/adept_gate_analysis.py" \
OBJECTIVE_ANALYSIS="scripts/analysis/adept_objective_analysis.py" \
PYTHON_BIN="$PWD/.venv/bin/python" \
OUTPUT_ROOT="$PWD/results/adept_point2_controls" \
MAX_PARALLEL=4 GPU_ID=0 \
  bash scripts/experiments/run_adept_point2_controls.sh
```

The command sets paths relative to this checkout and writes outputs under
`results/adept_point2_controls/`; adjust `OUTPUT_ROOT` to change it. Choose
`MAX_PARALLEL` according to available GPU and system memory.

</details>

## Outputs

Experiment outputs are written locally to the chosen output directory.

| Artifact | Contents |
|:--|:--|
| `*_Results.csv` | Global masked MAE and MSE for a run. |
| `*_TrainingLog.csv` | Training and validation history. |
| `*_DetailLog.csv` | Per-entity or per-window evaluation details. |
| `*_GateLog.csv` | Optional per-query gate weights, expert values, and errors. |
| `chronoa_summary.csv` | Aggregate summary used by analysis scripts. |

Query-level logs may contain sensitive information when using restricted data;
follow the relevant dataset's access and sharing rules.

## Repository structure

```text
Chronoa/
├── src/adept/                  # Model, data adapter, and training implementation
├── scripts/experiments/        # Forecasting and control runners
├── scripts/analysis/           # Metrics, tables, and gate analysis
├── revision/                   # Revision experiment and analysis runners
├── vendor/upstream_benchmark/  # Benchmark code and configurations
├── pyproject.toml              # Package and dependency configuration
└── README.md
```

See [`src/adept/README.md`](src/adept/README.md) for model internals and the
package map. The local manuscript folder (`docs/`), benchmark data, checkpoints,
logs, and generated results are excluded from Git.
