#!/usr/bin/env python3
"""Train and evaluate one Warpformer seed with consistent GPU transfers."""
from __future__ import annotations
import os
import random
import sys
import types
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
VENDOR=ROOT/'vendor'/'upstream_benchmark'
sys.path.insert(0,str(VENDOR))
from utils.configs import configs

index=int(os.environ['REVIEWER2_SEED_INDEX'])
tag=os.environ['REVIEWER2_TRAIN_TAG']
if index not in (0,1,2): raise ValueError('Expected seed index 0, 1, or 2')
if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for this launcher')
configs.use_gpu=True
configs.itr_i=index
configs.subfolder_train=tag
seed=2024+index
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
# The checked-in benchmark currently puts train/validation batches on CPU but
# test batches on CUDA. Correct just those three transfers in an in-memory copy;
# no vendor file is modified and all model/hyperparameter code stays unchanged.
path=VENDOR/'exp'/'exp_main.py'
source=path.read_text()
changes={
 'model_train = model_train.to("cpu")':'model_train = model_train.to(f"cuda:{self.configs.gpu_id}")',
 'batch = {k: v.to("cpu") for k, v in batch.items()}':'batch = {k: v.to(f"cuda:{self.configs.gpu_id}") for k, v in batch.items()}',
 'batch = {k: v.to(f"cpu:{self.configs.gpu_id}") for k, v in batch.items()}':'batch = {k: v.to(f"cuda:{self.configs.gpu_id}") for k, v in batch.items()}',
}
for old,new in changes.items():
 if source.count(old)!=1: raise RuntimeError(f'Unexpected benchmark device code for {old!r}')
 source=source.replace(old,new,1)
module=types.ModuleType('reviewer2_exp_main');module.__file__=str(path)
exec(compile(source,str(path),'exec'),module.__dict__)
print(f'Training Warpformer seed {seed}; checkpoint group {tag}',flush=True)
experiment=module.Exp_Main(configs)
experiment.train()
checkpoint=Path(configs.checkpoints)/configs.dataset_name/configs.model_name/configs.model_id/f'{configs.seq_len}_{configs.pred_len}'/tag/f'iter{index}'
if not checkpoint.is_dir() or not any(p.name in ('pytorch_model.bin','model.safetensors') for p in checkpoint.iterdir()):
 raise RuntimeError(f'Training finished without a model checkpoint: {checkpoint}')
configs.checkpoints_test=str(checkpoint)
experiment.test()
evaluations=[p for p in checkpoint.glob('eval_*') if all((p/n).is_file() for n in ('input_y.npy','input_y_mask.npy','input_sample_ID.npy','output_pred.npy','eval_configs.yaml'))]
if len(evaluations)!=1: raise RuntimeError(f'Expected one prediction export under {checkpoint}; found {len(evaluations)}')
print(f'Completed Warpformer seed {seed}: {evaluations[0]}',flush=True)
