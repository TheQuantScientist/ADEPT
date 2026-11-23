#!/usr/bin/env python3
"""Run reviewer 2 checkpoint inference and Warpformer seeds concurrently."""
from __future__ import annotations
import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
VENDOR=ROOT/'vendor'/'upstream_benchmark'
ANALYSIS=ROOT/'revision'/'reviewer2'/'run_reviewer2.py'
WARP=ROOT/'revision'/'reviewer2'/'run_warpformer_seed.py'
DATASETS=('P12','MIMIC','USHCN','HumanActivity')
SEEDS=(2024,2025,2026)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inference-workers',type=int,default=6)
    args=parser.parse_args()
    if args.inference_workers<1:parser.error('--inference-workers must be positive')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    logs=output/'worker_logs';logs.mkdir(exist_ok=True)
    warp_root=output/'warpformer'
    if warp_root.exists() and any(warp_root.iterdir()):
        raise RuntimeError('Warpformer output already exists. Use a new output directory to avoid ambiguous checkpoints.')
    tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m%d_%H%M')
    env=os.environ.copy();env.setdefault('OMP_NUM_THREADS','1');env.setdefault('MKL_NUM_THREADS','1')
    infer_queue=[(d,s) for d in DATASETS for s in SEEDS]
    active={};failed=[];stopping=False
    def stop(sig,_frame):
        nonlocal stopping
        stopping=True
        print(f'Received signal {sig}; terminating children',flush=True)
        for proc,_,log in active.values():proc.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(name,cmd,cwd,extra=None):
        log_path=logs/f'{name}.log';log=log_path.open('w',buffering=1)
        child_env=env.copy();child_env.update(extra or {})
        p=subprocess.Popen(cmd,cwd=cwd,env=child_env,stdout=log,stderr=subprocess.STDOUT)
        active[p.pid]=(p,name,log)
        print(f'Started {name}, pid={p.pid}, log={log_path}',flush=True)
    common=[sys.executable,'-u',str(WARP),'--gpu_id','0','--is_training','1','--loss','MSE',
        '--model_name','Warpformer','--model_id','reviewer2_warpformer_mimic',
        '--dataset_name','MIMIC_III','--dataset_root_path','storage/datasets/MIMIC_III',
        '--features','M','--seq_len','72','--pred_len','3','--enc_in','96','--c_out','96',
        '--d_model','256','--n_heads','4','--dropout','0','--n_layers','2','--use_multi_gpu','0',
        '--train_epochs','300','--patience','5','--val_interval','1','--itr','3',
        '--batch_size','16','--learning_rate','0.001','--num_workers','0',
        '--save_prediction_arrays','1','--checkpoints',str(warp_root)]
    launch('paired_initial',[sys.executable,'-u',str(ANALYSIS),'--mode','paired','--skip-mimic','--output',str(output)],ROOT)
    for index in range(3):
        launch(f'warpformer_seed{2024+index}',common,VENDOR,
               {'REVIEWER2_SEED_INDEX':str(index),'REVIEWER2_TRAIN_TAG':tag})
    while infer_queue or active:
        count=sum(name.startswith('inference_') for _,name,_ in active.values())
        while infer_queue and count<args.inference_workers and not stopping and not failed:
            dataset,seed=infer_queue.pop(0)
            cmd=[sys.executable,'-u',str(ANALYSIS),'--mode','inference','--device','cuda',
                 '--datasets',dataset,'--seeds',str(seed),'--sharded','--output',str(output)]
            launch(f'inference_{dataset}_seed{seed}',cmd,ROOT);count+=1
        for pid,(proc,name,log) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            log.close();del active[pid]
            print(f'Finished {name}, exit={code}',flush=True)
            if code:failed.append((name,code))
        if (failed or stopping) and not active:break
        if active:time.sleep(5)
    if stopping:return 130
    if failed:
        print(f'Worker failures: {failed}. Inspect worker logs; incomplete outputs were retained.',flush=True)
        return 1
    for mode in ('aggregate','paired'):
        print(f'Running final {mode}',flush=True)
        code=subprocess.call([sys.executable,'-u',str(ANALYSIS),'--mode',mode,'--output',str(output)],cwd=ROOT,env=env)
        if code:return code
    print(f'All reviewer 2 tasks completed. Results: {output}',flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
