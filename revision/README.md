# Reviewer 1 experiment runner

Run from the Chronoa directory in the existing project environment:

```bash
nohup /home/nckh2/qa/.venv/bin/python -u revision/run_reviewer1.py --device cuda > revision/reviewer1.log 2>&1 &
```

If the environment uses another Python executable, substitute it. The command runs 10 neural variants for each of four datasets and three seeds, sequentially; it can take substantial GPU time. It does not use existing `results/` outputs. All artifacts are written under `revision/results/`. Reissuing the command resumes completed neural variants, but recomputes the inexpensive baseline-selection and baseline-prediction outputs. `--overwrite` retrains all variants. Use `--datasets P12` or `--seeds 2024` to limit scope, or `--device cpu` when necessary.

The full model and all neural controls use the same mixed objective, validation checkpoint score, benchmark splits, and published training defaults. `model.pt` contains the chosen state dictionary, configuration, and selected epoch. `predictions.csv` contains query-level predictions and diagnostics; `training.csv` records validation history. `complete.json` is written after evaluation and parameter counting. Incomplete runs are retrained on resume.

Variants: `drop_*` removes the specified scalar anchor from the gate and residual and masks its output; dropping graph also replaces graph-based fallbacks with the global anchor. `event_only` retains the neural event forecast but removes expert mixing and residual. `expert_only` computes five scalar anchors and uses a gate and residual based only on those anchors, same-variable availability, and query gap; the graph anchor still uses trainable variable affinities. `fixed_gate` learns a single six-component weight vector during training, though its residual remains query-dependent. `neural_capacity` replaces the five anchors with five learned neural outputs while retaining the same decoder, gate and residual. Its parameter count is approximate rather than exact; inspect `parameter_counts.csv` before claiming matching. Counts distinguish allocated parameters from parameters reached by the loss; FLOPs are not measured.

Baselines use exact latest value or irregular-time exponentially weighted mean / clipped trend, with an entity-wide mean fallback when the target variable is absent. EMA/trend time scales are selected by validation MSE + 0.25 MAE over the recorded grid; no test targets are used in selection. All baseline predictions are clipped to the same scaled range as ADEPT's scalar anchors.

Full-ADEPT `cases.json` contains lowest, median, and highest error examples within each observed top-gate category, including actual input histories. These are descriptive, test-selected examples, not a representative sample or clinical explanation. `case_interventions.csv` records the effect of removing the latest same-variable event for those cases with the same trained model. Histories and entity identifiers may be sensitive; handle and share according to dataset access rules.

`metrics_by_seed.csv`, `metrics_summary.csv`, `paired_differences.csv`, `parameter_counts.csv`, and `gates_by_*_and_seed.csv` are written at successful completion. Paired differences are variant minus ADEPT and include target-weighted bootstrap intervals over seed-entity units; these intervals are descriptive and the seeds for the same entity are not independent entities. The runner checks target alignment before pairing. Gate summaries describe model weights, not causal evidence. The results may support negative or inconclusive conclusions.

For parallel dataset/seed execution, use the scheduler instead of running multiple copies of the sequential command:

```bash
nohup /home/nckh2/qa/.venv/bin/python -u revision/run_parallel.py --workers 4 --device cuda > revision/parallel.log 2>&1 &
```

Each worker has a separate log under `revision/worker_logs/`. The scheduler aggregates only after all workers succeed. Four is a starting limit, not an automatically tuned GPU-memory limit; change `--workers` after checking peak GPU and host memory, and do not run a second scheduler on the same output directory. If a worker fails, inspect its log and rerun the scheduler to resume completed variants.
