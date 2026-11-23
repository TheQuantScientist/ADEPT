#!/usr/bin/env python3
"""Run separate dataset/seed revision jobs concurrently and aggregate once."""
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
DATASET_NAMES = {'p12': 'P12', 'mimic': 'MIMIC', 'ushcn': 'USHCN',
                 'humanactivity': 'HumanActivity'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--datasets', nargs='+', default=['P12', 'MIMIC', 'USHCN', 'HumanActivity'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[2024, 2025, 2026])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, default=ROOT / 'revision' / 'results')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    log_dir = ROOT / 'revision' / 'worker_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for dataset in args.datasets:
        name = DATASET_NAMES.get(dataset.lower(), dataset)
        for seed in args.seeds:
            if all((args.output / name / variant / f'seed{seed}' / 'complete.json').is_file()
                   for variant in VARIANTS):
                print(f'Skip fully completed {name} seed {seed}', flush=True)
            else:
                tasks.append((dataset, seed))
    active = {}
    failures = []
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    stopping = False
    def stop(signum, _frame):
        nonlocal stopping
        print(f'Received signal {signum}; terminating workers', flush=True)
        stopping = True
        for process, _, _, _ in active.values():
            process.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while tasks or active:
        while tasks and len(active) < args.workers and not stopping and not failures:
            dataset, seed = tasks.pop(0)
            log_path = log_dir / f'{dataset}_seed{seed}.log'
            log = log_path.open('a', buffering=1)
            command = [sys.executable, '-u', str(RUNNER), '--datasets', dataset,
                       '--seeds', str(seed), '--device', args.device, '--output',
                       str(args.output), '--no-summary']
            print(f'Starting {dataset} seed {seed}: {log_path}', flush=True)
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
            active[proc.pid] = (proc, dataset, seed, log)
        for pid, (proc, dataset, seed, log) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del active[pid]
            print(f'Finished {dataset} seed {seed}, exit={code}', flush=True)
            if code:
                failures.append((dataset, seed, code))
        if (stopping or failures) and not active:
            break
        if active:
            time.sleep(5)
    if stopping:
        return 130
    if failures:
        print(f'Failed jobs: {failures}. Inspect worker logs; rerun to resume.', flush=True)
        return 1
    command = [sys.executable, '-u', str(RUNNER), '--output', str(args.output), '--summary-only']
    print('Aggregating completed runs', flush=True)
    return subprocess.call(command, cwd=ROOT, env=env)

if __name__ == '__main__':
    sys.exit(main())
