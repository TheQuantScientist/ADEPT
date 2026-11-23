#!/usr/bin/env python3
"""Reviewer 1 experiments. No existing results are read or overwritten."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from adept.experiments import chronoa_forecasting as core

EXPERTS = core.SCALAR_EXPERT_NAMES
NAMES = core.GATE_NAMES
VARIANTS = ['adept', 'drop_same_last', 'drop_same_ema', 'drop_trend',
            'drop_graph_ema', 'drop_global_ema', 'fixed_gate',
            'expert_only', 'event_only', 'neural_capacity']


class RevisionModel(core.ChronoaEventGraphNet):
    """All controls share the released ADEPT loss and validation selection."""
    def __init__(self, *args, revision_variant='adept', **kwargs):
        super().__init__(*args, **kwargs)
        self.revision_variant = revision_variant
        # Expert-only gate has no event encoding, retrieval, graph query context,
        # or neural forecast as input. The graph scalar uses learned variable affinities.
        self.expert_only_gate = nn.Sequential(nn.LayerNorm(7), nn.Linear(7, self.d_model),
                                              nn.GELU(), nn.Linear(self.d_model, 5))
        self.expert_only_residual = nn.Sequential(nn.LayerNorm(7), nn.Linear(7, self.d_model),
                                                  nn.GELU(), nn.Linear(self.d_model, 1))
        self.fixed_logits = nn.Parameter(torch.zeros(6))
        # Five learned alternatives to scalar anchors; same gate and residual topology.
        self.neural_anchors = nn.Linear(self.d_model, 5)

    def forward(self, batch, return_diagnostics=False):
        variant = self.revision_variant
        last_time = torch.max(batch.event_times.masked_fill(~batch.event_mask, -1e9), dim=1).values
        gap = (batch.query_times - last_time.unsqueeze(1)).unsqueeze(-1)
        experts, same_has = self.scalar_experts(batch)
        if variant == 'drop_graph_ema':
            # The released same-variable experts fall back to graph_ema. Replace
            # that fallback too, so the removed graph expert cannot leak back in.
            available = same_has.bool()
            experts = torch.cat([torch.where(available, experts[..., :3],
                                            experts[..., 4:5].expand_as(experts[..., :3])),
                                 experts[..., 3:]], dim=-1)
        if variant == 'expert_only':
            inputs = torch.cat([experts, same_has, gap], dim=-1)
            weights5 = torch.softmax(self.expert_only_gate(inputs), dim=-1)
            weights = torch.cat([torch.zeros_like(weights5[..., :1]), weights5], dim=-1)
            neural = torch.zeros_like(experts[..., 0])
            residual = .05 * torch.tanh(self.expert_only_residual(inputs)).squeeze(-1)
        else:
            memory = self.encode_events(batch.event_times, batch.event_vars, batch.event_values, batch.event_mask)
            queries = self.build_queries(batch.query_times, batch.query_vars)
            attention, _ = self.cross_attention(queries, memory, memory, key_padding_mask=~batch.event_mask, need_weights=False)
            kernel = self.delta_kernel_context(memory, batch.event_times, batch.event_vars, batch.event_mask,
                                               batch.query_times, batch.query_vars)
            hidden = self.decoder_backbone(torch.cat([queries, attention, kernel,
                                                       self.graph_context(batch.query_vars), self.gap_projection(gap)], dim=-1))
            neural = self.decoder_head(hidden).squeeze(-1)
            if variant == 'neural_capacity':
                experts = self.neural_anchors(hidden)
            if variant.startswith('drop_'):
                idx = EXPERTS.index(variant[5:])
                # Prevent the removed anchor from reaching either the gate or residual.
                mask = torch.ones_like(experts)
                mask[..., idx] = 0
                experts = experts * mask
            inputs = torch.cat([hidden, experts, same_has, gap], dim=-1)
            logits = self.expert_gate(inputs)
            if variant.startswith('drop_'):
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask[..., 1 + EXPERTS.index(variant[5:])] = False
                logits = logits.masked_fill(~mask, -1e9)
            if variant == 'fixed_gate':
                logits = self.fixed_logits.expand_as(logits)
            weights = torch.softmax(logits, dim=-1)
            residual = .05 * torch.tanh(self.expert_residual(inputs)).squeeze(-1)
            if variant == 'event_only':
                weights = torch.cat([torch.ones_like(neural.unsqueeze(-1)), torch.zeros_like(experts)], dim=-1)
                # Event-only baseline excludes the expert-conditioned residual.
                residual = torch.zeros_like(neural)
        pred = (weights * torch.cat([neural.unsqueeze(-1), experts], dim=-1)).sum(-1) + residual
        pred = pred.masked_fill(~batch.query_mask, 0)
        diag = {'gate_weights': weights, 'experts': experts, 'neural_decoded': neural,
                'residual': residual, 'same_has': same_has, 'gap': gap}
        return (pred, diag) if return_diagnostics else pred


def baseline(batch, kind, scale):
    """Irregular-time weighted baselines; missing variable falls back to global mean."""
    valid = batch.event_mask.unsqueeze(1)
    same = valid & (batch.query_vars.unsqueeze(-1) == batch.event_vars.unsqueeze(1))
    times = batch.event_times.unsqueeze(1)
    values = batch.event_values.unsqueeze(1)
    delta = (batch.query_times.unsqueeze(-1) - times).abs()
    have = same.any(-1)
    fallback = (values * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    last_idx = times.expand_as(same).masked_fill(~same, -1e9).argmax(-1)
    last = values.expand_as(same).gather(-1, last_idx.unsqueeze(-1)).squeeze(-1)
    last_t = times.expand_as(same).gather(-1, last_idx.unsqueeze(-1)).squeeze(-1)
    if kind == 'last':
        pred = last
    else:
        weights = core.ChronoaEventGraphNet.masked_softmax(-delta / scale, same)
        mean = (weights * values).sum(-1)
        if kind == 'ema':
            pred = mean
        else:
            mean_t = (weights * times).sum(-1)
            dt = times - mean_t.unsqueeze(-1)
            slope = (weights * dt * (values - mean.unsqueeze(-1))).sum(-1) / (weights * dt.square()).sum(-1).clamp_min(1e-4)
            slope = slope.clamp(-8, 8)
            pred = last + slope * (batch.query_times - last_t)
    return torch.where(have, pred, fallback).clamp(-12, 12).masked_fill(~batch.query_mask, 0)


def selected_baselines(samples, config, span, device):
    scales = [.01, .03, .1, .3, 1., 3.]
    records = []
    for kind in ('last', 'ema', 'trend'):
        for scale in ([.1] if kind == 'last' else scales):
            absolute = square = count = 0
            with torch.no_grad():
                for _, b in core.iter_batches(samples, config.batch_size, span, device, False, config.random_seed):
                    error = (baseline(b, kind, scale) - b.query_targets)[b.query_mask]
                    absolute += error.abs().sum().item(); square += error.square().sum().item(); count += error.numel()
            records.append(dict(baseline=kind, scale=scale, mae=absolute/count, mse=square/count,
                                score=square/count + .25*absolute/count, count=count))
    frame = pd.DataFrame(records)
    chosen = frame.sort_values(['baseline', 'score', 'scale']).groupby('baseline', sort=False).first().reset_index()
    return frame, chosen


def rows_for_predictions(samples, predictions, benchmark, variant, seed, diagnostics=None):
    rows = []
    for si, (sample, pred) in enumerate(zip(samples, predictions)):
        diag = diagnostics[si] if diagnostics is not None else None
        for q, p in enumerate(pred):
            y = float(sample.query_targets[q]); err = float(p)-y
            row = dict(dataset=benchmark.name, variant=variant, seed=seed, sample_index=si,
                       entity=str(sample.key), variable=benchmark.columns[int(sample.query_vars[q])],
                       query_index=q, step=int(sample.query_steps[q]), target_time=float(sample.query_times[q]),
                       actual=y, prediction=float(p), ae=abs(err), se=err*err)
            if diag:
                row.update(neural=float(diag['neural_decoded'][q]), residual=float(diag['residual'][q]),
                           same_available=int(diag['same_has'][q][0]), gap=float(diag['gap'][q][0]))
                for i, name in enumerate(NAMES): row['gate_'+name] = float(diag['gate_weights'][q][i])
                for i, name in enumerate(EXPERTS): row['expert_'+name] = float(diag['experts'][q][i])
                row['top'] = NAMES[int(np.argmax(diag['gate_weights'][q]))]
            rows.append(row)
    return pd.DataFrame(rows)


def export_cases(frame, samples, benchmark, out):
    # Deterministic extreme-error and typical examples for every observed top gate.
    chosen = []
    for _, group in frame.groupby('top', sort=True):
        ordered = group.sort_values(['ae', 'sample_index', 'query_index'])
        chosen.extend([ordered.iloc[0], ordered.iloc[len(ordered)//2], ordered.iloc[-1]])
    cases = []
    for row in chosen:
        sample = samples[int(row.sample_index)]
        case = row.to_dict()
        case['history'] = [dict(time=float(t), variable=benchmark.columns[int(v)], value=float(x))
                           for t, v, x in zip(sample.event_times, sample.event_vars, sample.event_values)]
        cases.append(case)
    with (out / 'cases.json').open('w') as f: json.dump(cases, f, indent=2, allow_nan=False)
    return chosen


def export_interventions(chosen, samples, model, benchmark, config, device, out):
    """Remove the most recent same-variable input from selected cases."""
    rows = []
    model.eval()
    for row in chosen:
        sample = samples[int(row.sample_index)]
        q = int(row.query_index)
        matching = np.flatnonzero(sample.event_vars == sample.query_vars[q])
        if len(matching) == 0 or len(sample.event_values) <= 1:
            rows.append(dict(sample_index=int(row.sample_index), query_index=q,
                             status='no_removable_same_variable_event'))
            continue
        removed = int(matching[np.argmax(sample.event_times[matching])])
        keep = np.arange(len(sample.event_values)) != removed
        modified = copy.copy(sample)
        modified.event_times = sample.event_times[keep]
        modified.event_vars = sample.event_vars[keep]
        modified.event_values = sample.event_values[keep]
        with torch.no_grad():
            batch = core.collate_batch([modified], benchmark.lookback_span, device)
            prediction, diag = model(batch, return_diagnostics=True)
        entry = dict(sample_index=int(row.sample_index), query_index=q, status='ok',
                     removed_time=float(sample.event_times[removed]),
                     removed_value=float(sample.event_values[removed]),
                     original_prediction=float(row.prediction),
                     intervened_prediction=float(prediction[0,q].item()),
                     actual=float(row.actual))
        for i, name in enumerate(NAMES):
            entry['gate_before_'+name] = float(row['gate_'+name])
            entry['gate_after_'+name] = float(diag['gate_weights'][0,q,i].item())
        rows.append(entry)
    pd.DataFrame(rows).to_csv(out/'case_interventions.csv', index=False)


def summarize(out, seed):
    frames = []
    for p in sorted(out.glob('*/*/seed*/predictions.csv')):
        frame = pd.read_csv(p); frames.append(frame)
    if not frames: return
    allrows = pd.concat(frames, ignore_index=True)
    summary = allrows.groupby(['dataset', 'variant', 'seed'], as_index=False).agg(
        mae=('ae', 'mean'), mse=('se', 'mean'), targets=('ae', 'size'))
    summary.to_csv(out / 'metrics_by_seed.csv', index=False)
    summary.groupby(['dataset', 'variant'], as_index=False).agg(
        mae_mean=('mae','mean'), mae_std=('mae','std'), mse_mean=('mse','mean'),
        mse_std=('mse','std'), seeds=('seed','nunique')).to_csv(out / 'metrics_summary.csv', index=False)
    key = ['dataset','seed','sample_index','query_index','entity','variable','step','target_time']
    reference = allrows[allrows.variant.eq('adept')][key+['actual','ae','se']]
    pairs = []
    rng = np.random.default_rng(20261121)
    for variant, group in allrows[~allrows.variant.eq('adept')].groupby('variant'):
        matched = group.merge(reference, on=key, suffixes=('_variant','_adept'), validate='one_to_one')
        if len(matched) != len(group) or not np.allclose(matched.actual_variant, matched.actual_adept, atol=1e-6):
            raise ValueError(f'Unmatched or unequal targets in {variant}; inspect query exports')
        for dataset, block in matched.groupby('dataset'):
            block = block.copy()
            block['d_mae'] = block.ae_variant - block.ae_adept
            block['d_mse'] = block.se_variant - block.se_adept
            units = block.groupby(['seed','entity'], sort=False).agg(
                sum_mae=('d_mae','sum'), sum_mse=('d_mse','sum'), n=('d_mae','size'))
            values = units.to_numpy(dtype=float)
            indices = rng.integers(0, len(values), size=(1000, len(values)))
            boot = values[indices].sum(axis=1)
            mae_ci = np.quantile(boot[:,0] / boot[:,2], [.025,.975])
            mse_ci = np.quantile(boot[:,1] / boot[:,2], [.025,.975])
            pairs.append(dict(dataset=dataset,variant=variant,targets=len(block),
                              seed_entity_units=len(units),
                              delta_mae_variant_minus_adept=block.d_mae.mean(),
                              mae_ci_low=mae_ci[0], mae_ci_high=mae_ci[1],
                              delta_mse_variant_minus_adept=block.d_mse.mean(),
                              mse_ci_low=mse_ci[0], mse_ci_high=mse_ci[1]))
    pd.DataFrame(pairs).to_csv(out / 'paired_differences.csv', index=False)
    metadata = [json.loads(path.read_text()) for path in sorted(out.glob('*/*/seed*/complete.json'))]
    if metadata:
        pd.DataFrame(metadata).to_csv(out / 'parameter_counts.csv', index=False)
    gates = allrows[allrows.variant.eq('adept')].copy()
    if 'gap' in gates:
        gates['gap_bin'] = pd.cut(gates.gap.abs(), [-np.inf,.05,.25,.5,1.,np.inf])
        for col in ['variable', 'same_available', 'gap_bin']:
            gates.groupby(['dataset','seed',col], observed=True, dropna=False).agg(
                queries=('ae','size'), mae=('ae','mean'), **{n:('gate_'+n,'mean') for n in NAMES}
            ).reset_index().to_csv(out / f'gates_by_{col}_and_seed.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['P12','MIMIC','USHCN','HumanActivity'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[2024,2025,2026])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=16)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--output', type=Path, default=ROOT/'revision'/'results')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--no-summary', action='store_true', help='Leave aggregation to the parallel launcher.')
    ap.add_argument('--summary-only', action='store_true', help='Aggregate existing outputs without training.')
    ap.add_argument('--variant', choices=VARIANTS, help='Run one neural variant only.')
    ap.add_argument('--skip-baselines', action='store_true', help='Reuse previously exported baseline predictions.')
    ap.add_argument('--baseline-only', action='store_true', help='Evaluate standalone baselines without training a neural variant.')
    args = ap.parse_args()
    if args.baseline_only and (args.skip_baselines or args.variant):
        ap.error('--baseline-only cannot be combined with --skip-baselines or --variant')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        summarize(args.output, args.seeds[0])
        return
    # Serialize duplicate dataset/seed jobs, including manually started jobs and
    # jobs subsequently picked up by the scheduler. The OS releases this lock
    # even if a worker exits unexpectedly.
    job_lock = None
    if len(args.datasets) == 1 and len(args.seeds) == 1:
        lock_dir = args.output / '_locks'
        lock_dir.mkdir(parents=True, exist_ok=True)
        suffix = '_' + args.variant if args.variant else ('_baselines' if args.baseline_only else '')
        lock_name = f'{core.canonical_dataset_name(args.datasets[0])}_{args.seeds[0]}{suffix}.lock'
        job_lock = (lock_dir / lock_name).open('w')
        print(f'Waiting for dataset/seed lock: {lock_name}', flush=True)
        fcntl.flock(job_lock, fcntl.LOCK_EX)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger('revision')
    for dataset in args.datasets:
        for seed in args.seeds:
            print(f'Loading {dataset}, seed {seed}', flush=True)
            config = core.ChronoaConfig(dataset_name=dataset, architecture_ablation='apex', random_seed=seed,
                output_dir=args.output/'_metadata'/dataset/str(seed), epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size, learning_rate=.0006, weight_decay=.0001, d_model=96, layers=2,
                heads=4, dropout=.08, delta_kernels=6, max_events=2048, max_queries_per_sample=768, device=args.device)
            device = core.choose_device(config)
            benchmark = core.load_benchmark_data(config)
            def load(ds, training, label):
                return core.build_neural_samples(ds, benchmark, config, training, False, None, logger, label)
            train = load(benchmark.fit_dataset, True, 'train')
            val = load(benchmark.selection_dataset, False, 'validation')
            test = load(benchmark.test_dataset, False, 'test')
            if not train or not val or not test: raise RuntimeError('One or more data splits are empty')
            dataset_dir = args.output / benchmark.name
            if not args.skip_baselines:
                candidates, chosen = selected_baselines(val, config, benchmark.lookback_span, device)
                base_dir = dataset_dir/'baseline_selection'/f'seed{seed}'; base_dir.mkdir(parents=True,exist_ok=True)
                candidates.to_csv(base_dir/'validation_grid.csv', index=False); chosen.to_csv(base_dir/'selected.csv', index=False)
                for _, item in chosen.iterrows():
                    name='baseline_'+item.baseline; out=dataset_dir/name/f'seed{seed}'; out.mkdir(parents=True,exist_ok=True)
                    predictions=[]
                    with torch.no_grad():
                        for samples,b in core.iter_batches(test,config.batch_size,benchmark.lookback_span,device,False,seed):
                            p=baseline(b,item.baseline,float(item.scale)).cpu().numpy()
                            predictions.extend(p[i,:len(s.query_targets)] for i,s in enumerate(samples))
                    rows_for_predictions(test,predictions,benchmark,name,seed).to_csv(out/'predictions.csv',index=False)
            for variant in ([] if args.baseline_only else ([args.variant] if args.variant else VARIANTS)):
                out=dataset_dir/variant/f'seed{seed}'; out.mkdir(parents=True,exist_ok=True)
                if (out/'complete.json').exists() and not args.overwrite:
                    print(f'Skip completed {benchmark.name} {seed} {variant}',flush=True); continue
                print(f'Training {benchmark.name} {seed} {variant}',flush=True)
                original_cls=core.ChronoaEventGraphNet
                class SelectedModel(RevisionModel):
                    def __init__(self,*a,**kw): super().__init__(*a,revision_variant=variant,**kw)
                core.ChronoaEventGraphNet=SelectedModel
                try: model, log=core.train_neural_model(train,val,benchmark,config,device,logger)
                finally: core.ChronoaEventGraphNet=original_cls
                log.to_csv(out/'training.csv',index=False)
                torch.save({'state_dict': model.state_dict(), 'variant':variant, 'dataset':benchmark.name,
                            'seed':seed, 'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(config).items()},
                            'best_epoch':int(log.loc[log.Validation_score.idxmin(),'Epoch'])}, out/'model.pt')
                predictions, diag=core.predict_samples_with_gate_diagnostics(model,test,config,benchmark.lookback_span,device)
                frame=rows_for_predictions(test,predictions,benchmark,variant,seed,diag)
                frame.to_csv(out/'predictions.csv',index=False)
                if variant=='adept':
                    cases = export_cases(frame,test,benchmark,out)
                    export_interventions(cases,test,model,benchmark,config,device,out)
                # Count parameters touched by the loss for an actual sample, in addition to allocated size.
                model.train(); model.zero_grad(set_to_none=True)
                _, b=next(iter(core.iter_batches(train,1,benchmark.lookback_span,device,False,seed)))
                loss=core.masked_loss(model(b),b.query_targets,b.query_mask,architecture='apex')
                loss.backward()
                active=sum(p.numel() for p in model.parameters() if p.grad is not None)
                total=sum(p.numel() for p in model.parameters())
                info=dict(dataset=benchmark.name,variant=variant,seed=seed,trainable_allocated=total,
                          trainable_used_in_forward=active,best_epoch=int(log.loc[log.Validation_score.idxmin(),'Epoch']),
                          targets=len(frame),mae=float(frame.ae.mean()),mse=float(frame.se.mean()),
                          note='Active parameter count includes parameters with gradients; allocated includes unused control branches.')
                (out/'complete.json').write_text(json.dumps(info,indent=2))
                del model
                if device.type=='cuda': torch.cuda.empty_cache()
    if not args.no_summary:
        summarize(args.output,args.seeds[0])
    print('Finished. Results:', args.output, flush=True)

if __name__=='__main__': main()
