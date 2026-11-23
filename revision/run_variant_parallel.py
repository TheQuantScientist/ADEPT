#!/usr/bin/env python3
"""Run each dataset/seed/model variant as an independent revision job."""
from __future__ import annotations
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'revision' / 'run_reviewer1.py'
VARIANTS = ('adept', 'drop_same_last', 'drop_same_ema', 'drop_trend',
            'drop_graph_ema', 'drop_global_ema', 'fixed_gate', 'expert_only',
            'event_only', 'neural_capacity')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', default=['P12', 'MIMIC'], choices=['P12','MIMIC','USHCN','HumanActivity'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[2024,2025,2026])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / 'worker_logs'
    logs.mkdir(parents=True, exist_ok=True)
    tasks = []
    # Interleave datasets and seeds, so both start immediately.
    for variant in VARIANTS:
        for seed in args.seeds:
            for dataset in args.datasets:
                done = args.output / dataset / variant / f'seed{seed}' / 'complete.json'
                if not done.is_file():
                    tasks.append((dataset, seed, variant))
    for seed in args.seeds:
        for dataset in args.datasets:
            required = [args.output / dataset / f'baseline_{kind}' / f'seed{seed}' / 'predictions.csv'
                        for kind in ('last', 'ema', 'trend')]
            selected = args.output / dataset / 'baseline_selection' / f'seed{seed}' / 'selected.csv'
            if not all(p.is_file() and p.stat().st_size for p in [*required, selected]):
                tasks.append((dataset, seed, 'baselines'))
    print(f'Pending jobs: {len(tasks)}; worker limit: {args.workers}', flush=True)
    active = {}
    failures = []
    stopping = False
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    def stop(sig, _frame):
        nonlocal stopping
        stopping = True
        print(f'Received signal {sig}; terminating workers', flush=True)
        for process, _, log in active.values():
            process.terminate()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while tasks or active:
        while tasks and len(active) < args.workers and not stopping and not failures:
            dataset, seed, variant = tasks.pop(0)
            log_path = logs / f'{dataset}_seed{seed}_{variant}.log'
            log = log_path.open('a', buffering=1)
            cmd = [sys.executable, '-u', str(RUNNER), '--datasets', dataset, '--seeds', str(seed),
                   '--device', args.device, '--output', str(args.output), '--no-summary']
            cmd += ['--baseline-only'] if variant == 'baselines' else ['--variant', variant, '--skip-baselines']
            print(f'Start {dataset} {seed} {variant} -> {log_path}', flush=True)
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
            active[proc.pid] = (proc, (dataset, seed, variant), log)
        for pid, (proc, task, log) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del active[pid]
            print(f'Finished {task}, exit={code}', flush=True)
            if code:
                failures.append((task, code))
        if (stopping or failures) and not active:
            break
        if active:
            time.sleep(5)
    if stopping:
        return 130
    if failures:
        print(f'Failed jobs: {failures}. Inspect worker logs and rerun to resume.', flush=True)
        return 1
    print('Aggregating completed results', flush=True)
    return subprocess.call([sys.executable, '-u', str(RUNNER), '--output', str(args.output), '--summary-only'], cwd=ROOT, env=env)

if __name__ == '__main__':
    sys.exit(main())
