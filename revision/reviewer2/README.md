# Reviewer 2 validation and inference

From the Chronoa checkout, launch the parallel scheduler:

```bash
reviewer2_run_dir="revision/reviewer2/parallel_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$reviewer2_run_dir"
nohup /home/nckh2/qa/.venv/bin/python -u revision/reviewer2/run_parallel.py \
  --output "$PWD/$reviewer2_run_dir" --inference-workers 6 \
  > "$reviewer2_run_dir/parallel.log" 2>&1 &
echo "$reviewer2_run_dir"
```

The scheduler concurrently runs the initial paired analysis, three **new** MIMIC Warpformer seed jobs, and up to six of the twelve ADEPT checkpoint inference jobs. Each Warpformer seed trains and exports predictions independently. Each ADEPT job evaluates its checkpoint on the original and perturbed histories. The scheduler aggregates only after every worker completes. Each process has its own log under the chosen output directory's `worker_logs/`. Set `--inference-workers` to adjust parallelism; all jobs share the GPU and system memory. The Warpformer results are newly trained, not the original Table 1 checkpoints. Check whether their results, splits, masks, and configurations are appropriate before comparing or replacing paper values. `run_all.sh` is the earlier sequential launcher and is not needed for this workflow.

The paired analysis uses original standard ADEPT prediction logs and saved baseline arrays from the adjacent ChronoLM benchmark tree. It checks entity, variable, step, and target values before matching. P12 and HumanActivity have full common-target coverage in the initial audit; USHCN has 361 of 368 ADEPT targets per iteration, so results are computed and labeled on common support. Entity bootstrap intervals resample the same entities across the three model iterations; HumanActivity clusters windows by the subject identifier before the underscore. Only three HumanActivity subject clusters were present in the preliminary check, making its bootstrap interval unstable and unsuitable for strong population claims. Intervals average absolute errors over model iterations before clustering and do not quantify full training-run uncertainty.

The perturbation outputs use `revision/results/<dataset>/adept/seed*/model.pt`. These are separately trained revision-control checkpoints, not the lab's original Table 1 checkpoints. The runner checks that the original-history checkpoint predictions reproduce the stored control predictions before evaluating perturbations. The outputs are not evidence that the Table 1 model generated those perturbation results.

The initial paired analysis records MIMIC as pending in `paired_missing.json`. The final analysis reruns after training; it fails rather than silently pairing mismatched target values or ambiguous baseline outputs. If a command fails, inspect `parallel.log` and the worker logs. Use a new output directory for a fresh rerun; do not choose a favorable run after examining test results.

For the requested APN/GraFITi/tPatchGNN-only scope, use:

```bash
nohup bash revision/reviewer2/run_three_baselines.sh > revision/reviewer2/three_baselines.log 2>&1 &
```

This aggregates the existing twelve ADEPT checkpoint-evaluation jobs and computes the twelve ADEPT-versus-baseline paired analyses on CPU. No model is trained or evaluated again. Outputs go to `three_baselines_results/`, and the inference summary goes into the existing `parallel_20261122_083519/` folder. The program checks actual target values and reports target coverage. HumanActivity tPatchGNN uses verified within-entity/variable chronological ordering because padded step indices differ; the exported arrays do not contain explicit target timestamps, so this alignment assumption should be noted. Intervals cluster HumanActivity by subject (three clusters). USHCN tPatchGNN and P12/MIMIC tPatchGNN use shared target subsets. MIMIC tPatchGNN is not the strongest baseline in Table 1, which also includes Warpformer; restrict the scope of any claim accordingly.
