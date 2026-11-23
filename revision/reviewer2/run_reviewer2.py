#!/usr/bin/env python3
"""Reviewer 2: paired benchmark analysis and same-checkpoint perturbations."""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'revision'))
from adept.experiments import chronoa_forecasting as core
from run_reviewer1 import RevisionModel

SEEDS = (2024, 2025, 2026)
DATASETS = ('P12', 'MIMIC', 'USHCN', 'HumanActivity')
SPECS = {
 'P12': ('p12','GraFITi','sl36_pl3'),
 'MIMIC': ('mimic','Warpformer',None),
 'USHCN': ('ushcn','tPatchGNN','sl150_pl3'),
 'HumanActivity': ('humanactivity','APN','sl3000_pl300'),
}
CONDITIONS = [('original',1.0), ('shuffle_timestamps',1.0), ('swap_halves',1.0),
 ('drop_early_history',1.0), ('keep_last_obs',1.0), ('keep_time_gaps',1.0),
 ('original',.75), ('original',.5), ('original',.25)]


def standard_frame(dataset, seed):
    slug = SPECS[dataset][0]
    folder = PROJECT/'results'/'adept_seeded_parallel12'/f'seed{seed}_standard_{slug}'/slug
    files = list(folder.glob('*DetailLog.csv'))
    if len(files) != 1: raise RuntimeError(f'Expected one standard prediction file in {folder}, found {len(files)}')
    frame = pd.read_csv(files[0], dtype={'Entity': str, 'Variable': str})
    frame['Step'] = pd.to_numeric(frame.Step).astype(int)
    frame['Entity'] = frame.Entity.str.replace(r'\.0$', '', regex=True)
    keys=['Entity','Variable','Step']
    if frame.duplicated(keys).any(): raise ValueError(f'duplicate target coordinates: {files[0]}')
    return frame, files[0]


def baseline_dirs(dataset, output):
    slug,model,tag=SPECS[dataset]
    if dataset=='MIMIC':
        root=output/'warpformer'/'MIMIC_III'/'Warpformer'/'reviewer2_warpformer_mimic'/'72_3'
    else:
        root=Path('/home/nckh2/qa/ChronoLM/vendor/upstream_benchmark/storage/results/cross_baseline')/dataset/model
    if not root.exists(): return {}
    found={}
    for p in sorted(root.rglob('output_pred.npy')):
        if tag and tag not in str(p): continue
        d=p.parent
        if not all((d/x).is_file() for x in ('input_y.npy','input_y_mask.npy','input_sample_ID.npy')): continue
        cfg=d/'eval_configs.yaml'
        if not cfg.is_file(): continue
        import yaml
        obj=yaml.safe_load(cfg.read_text()) or {}
        if obj.get('history_perturbation','original')!='original' or float(obj.get('history_keep_fraction',1.0))!=1.0: continue
        it=next((part for part in p.parts if part.startswith('iter') and part[4:].isdigit()),None)
        if it is None: continue
        index=int(it[4:])
        if index in found: raise RuntimeError(f'Ambiguous baseline evaluations for {dataset} {it}: {found[index]} and {d}; select the intended source explicitly')
        found[index]=d
    return found


def variable_names(dataset):
    if dataset == 'MIMIC':
        # The per-variable metric CSV omits variables without test targets, so
        # it cannot preserve MIMIC's full 96-channel array indices.
        config=core.ChronoaConfig(dataset_name='MIMIC',architecture_ablation='apex',
            output_dir=PROJECT/'revision'/'reviewer2'/'metadata'/'mimic_mapping')
        return [str(x) for x in core.load_benchmark_data(config).columns]
    slug=SPECS[dataset][0]
    folder=PROJECT/'results'/'adept_seeded_parallel12'/f'seed2024_standard_{slug}'/slug
    path=next(folder.glob('*Checkpoint.csv'))
    with path.open(newline='') as f: return [row['Variable'] for row in csv.DictReader(f)]


def human_map():
    # Upstream array IDs index the exact test-window order, not subject IDs.
    config=core.ChronoaConfig(dataset_name='HumanActivity',architecture_ablation='apex',
        output_dir=PROJECT/'revision'/'reviewer2'/'metadata'/'human_mapping')
    benchmark=core.load_benchmark_data(config)
    return {str(i):str(sample.key) for i,sample in enumerate(benchmark.test_dataset)}


def baseline_frame(folder, dataset, names, mapping):
    y=np.load(folder/'input_y.npy',allow_pickle=False)
    pred=np.load(folder/'output_pred.npy',allow_pickle=False)
    mask=np.load(folder/'input_y_mask.npy',allow_pickle=False).astype(bool)
    ids=np.load(folder/'input_sample_ID.npy',allow_pickle=False).reshape(-1)
    if y.ndim!=3 or y.shape!=pred.shape or y.shape!=mask.shape or y.shape[0]!=len(ids):
        raise ValueError(f'Unexpected baseline array shapes in {folder}')
    if y.shape[-1]!=len(names): raise ValueError(f'Variable count mismatch in {folder}: {y.shape[-1]} versus {len(names)}')
    a,b,c=np.nonzero(mask)
    entities=[]
    for idx in a:
        val=str(int(ids[idx])) if float(ids[idx]).is_integer() else str(ids[idx])
        entities.append(mapping[val] if mapping is not None else val)
    frame=pd.DataFrame({'Entity':entities,'Variable':[names[i] for i in c], 'Step':b,
                        'baseline_actual':y[a,b,c], 'baseline_pred':pred[a,b,c]})
    if frame.duplicated(['Entity','Variable','Step']).any(): raise ValueError(f'Duplicate coordinates in {folder}')
    return frame


def paired(output, reps, seed, skip_mimic=False):
    output.mkdir(parents=True, exist_ok=True)
    rows=[]; missing=[]; rng=np.random.default_rng(seed); hm=None
    for dataset in DATASETS:
        if dataset=='MIMIC' and skip_mimic:
            missing.append({'dataset':dataset,'model':'Warpformer', 'reason':'Warpformer evaluation is still scheduled'})
            continue
        dirs=baseline_dirs(dataset,output)
        if not dirs:
            missing.append({'dataset':dataset,'model':SPECS[dataset][1], 'reason':'No baseline prediction arrays found'})
            print(f'Missing {dataset}/{SPECS[dataset][1]} prediction arrays',flush=True);continue
        if sorted(dirs)!=[0,1,2]: raise RuntimeError(f'Expected baseline iterations 0,1,2 for {dataset}, got {sorted(dirs)}')
        if dataset=='HumanActivity': hm=human_map()
        names=variable_names(dataset); merged=None; coverage=[]
        for j,seed_id in enumerate(SEEDS):
            af,source=standard_frame(dataset,seed_id)
            bf=baseline_frame(dirs[j],dataset,names,hm if dataset=='HumanActivity' else None)
            keys=['Entity','Variable','Step']; pair=af.merge(bf,on=keys,how='inner',validate='one_to_one')
            if pair.empty: raise ValueError(f'No common targets for {dataset} seed {seed_id}')
            if not np.allclose(pair.Actual_scaled,pair.baseline_actual,atol=2e-5,rtol=1e-4):
                raise ValueError(f'Mismatched targets for {dataset} seed {seed_id}')
            coverage.append((len(af),len(bf),len(pair)))
            z=pair[keys+['Actual_scaled']].copy()
            z[f'adept_ae_{j}']=(pair.Actual_scaled-pair.Predicted_scaled).abs()
            z[f'baseline_ae_{j}']=(pair.baseline_actual-pair.baseline_pred).abs()
            merged=z if merged is None else merged.merge(z,on=keys,suffixes=('',f'_{j}'),validate='one_to_one')
        if any(len(merged)!=x[2] for x in coverage): raise ValueError(f'Common support changed across seeds for {dataset}')
        for j in (1,2):
            if not np.allclose(merged.Actual_scaled,merged[f'Actual_scaled_{j}'],atol=2e-5,rtol=1e-4):
                raise ValueError(f'ADEPT targets changed across seeds for {dataset}')
        merged['adept_mean_ae']=merged[[f'adept_ae_{j}' for j in range(3)]].mean(axis=1)
        merged['baseline_mean_ae']=merged[[f'baseline_ae_{j}' for j in range(3)]].mean(axis=1)
        merged['delta_adept_minus_baseline']=merged.adept_mean_ae-merged.baseline_mean_ae
        # HumanActivity entity IDs are activity windows like E03_0. Cluster windows
        # belonging to the same subject to avoid treating one person as independent.
        merged['cluster']=merged.Entity.str.split('_').str[0] if dataset=='HumanActivity' else merged.Entity
        units=merged.groupby('cluster').agg(delta=('delta_adept_minus_baseline','sum'), n=('Step','size'))
        vals=units.to_numpy(float);n=len(vals);draws=np.empty(reps)
        for b in range(reps):
            draw=vals[rng.integers(0,n,size=n)].sum(axis=0);draws[b]=draw[0]/draw[1]
        row={'dataset':dataset,'baseline':SPECS[dataset][1], 'common_targets':len(merged),
             'adept_targets_per_seed':coverage[0][0],'baseline_targets_per_iteration':coverage[0][1],
             'adept_target_coverage':len(merged)/coverage[0][0],'clusters':n,'baseline_iterations':len(dirs),
             'adept_mae_common':merged.adept_mean_ae.mean(),'baseline_mae_common':merged.baseline_mean_ae.mean(),
             'delta_adept_minus_baseline':merged.delta_adept_minus_baseline.mean(),
             'ci_low':np.quantile(draws,.025),'ci_high':np.quantile(draws,.975),
             'cluster_unit':'subject' if dataset=='HumanActivity' else 'entity',
             'source_note':'new Warpformer run, not original Table 1 checkpoint' if dataset=='MIMIC' else 'saved ChronoLM cross-baseline arrays; verify original run provenance'}
        rows.append(row); merged.to_csv(output/f'paired_targets_{dataset}.csv',index=False)
        print(f'{dataset}: matched {len(merged)}/{coverage[0][0]} ADEPT targets, {n} clusters',flush=True)
    pd.DataFrame(rows).to_csv(output/'paired_baseline_ci.csv',index=False)
    (output/'paired_missing.json').write_text(json.dumps(missing,indent=2))


def inference(output, device_name, datasets, seeds, original_only=False, sharded=False):
    log=logging.getLogger('reviewer2'); logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    rows=[]
    for dataset in datasets:
        for seed in seeds:
            folder=PROJECT/'revision'/'results'/dataset/'adept'/f'seed{seed}'
            checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
            cfg=checkpoint['config']
            if checkpoint['dataset']!=dataset or int(checkpoint['seed'])!=seed or checkpoint['variant']!='adept':
                raise ValueError(f'Checkpoint identity mismatch in {folder}')
            base=core.ChronoaConfig(dataset_name=dataset,architecture_ablation='apex',random_seed=seed,
                output_dir=output/'metadata'/dataset/str(seed),seq_len=int(cfg['seq_len']),pred_len=int(cfg['pred_len']),
                batch_size=int(cfg['batch_size']),d_model=int(cfg['d_model']),layers=int(cfg['layers']),
                heads=int(cfg['heads']),dropout=float(cfg['dropout']),delta_kernels=int(cfg['delta_kernels']),
                max_events=int(cfg['max_events']),max_queries_per_sample=int(cfg['max_queries_per_sample']),device=device_name)
            benchmark=core.load_benchmark_data(base);device=core.choose_device(base)
            model=RevisionModel(n_variables=len(benchmark.columns),d_model=base.d_model,n_heads=base.heads,
                n_layers=base.layers,dropout=base.dropout,delta_kernels=base.delta_kernels,
                architecture_ablation='apex',revision_variant='adept').to(device)
            model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
            ref=pd.read_csv(folder/'predictions.csv',dtype={'entity':str,'variable':str})
            reference=None
            for condition,keep in (CONDITIONS[:1] if original_only else CONDITIONS):
                tag=condition if keep==1 else f'keep_{int(keep*100):03d}'
                config=replace(base,history_perturbation=condition,history_keep_fraction=keep)
                samples=core.build_neural_samples(benchmark.test_dataset,benchmark,config,False,
                    condition!='original' or keep!=1,None,log,f'{dataset}/{seed}/{tag}')
                predictions=core.predict_samples(model,samples,config,benchmark.lookback_span,device)
                details=[]
                for sample,pred in zip(samples,predictions):
                    for q,x in enumerate(pred):
                        y=float(sample.query_targets[q]);details.append({'entity':str(sample.key),
                            'variable':str(benchmark.columns[int(sample.query_vars[q])]), 'step':int(sample.query_steps[q]),
                            'actual':y,'prediction':float(x),'ae':abs(float(x)-y),'se':(float(x)-y)**2})
                frame=pd.DataFrame(details);keys=['entity','variable','step']
                if frame.duplicated(keys).any():raise ValueError(f'Duplicate target coordinates: {dataset}/{seed}/{tag}')
                if reference is None:
                    matched=frame.merge(ref,on=keys,suffixes=('_new','_saved'),validate='one_to_one')
                    if len(matched)!=len(ref) or len(matched)!=len(frame):raise ValueError(f'Original target coverage mismatch: {folder}')
                    if not np.allclose(matched.actual_new,matched.actual_saved,atol=2e-5,rtol=1e-4):raise ValueError(f'Original targets mismatch: {folder}')
                    maxdiff=float(np.max(np.abs(matched.prediction_new-matched.prediction_saved)))
                    if maxdiff>1e-3:raise ValueError(f'Original checkpoint predictions differ by {maxdiff}: {folder}')
                    print(f'{dataset} seed {seed}: checkpoint prediction max difference {maxdiff:.3g}',flush=True)
                    reference=frame
                pair=reference.merge(frame,on=keys,suffixes=('_original','_condition'),validate='one_to_one')
                if len(pair)!=len(reference) or len(pair)!=len(frame):raise ValueError(f'Target mismatch for {dataset}/{seed}/{tag}')
                if not np.allclose(pair.actual_original,pair.actual_condition,atol=1e-6):raise ValueError('Targets changed under perturbation')
                out=output/'same_checkpoint'/dataset/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
                frame.to_csv(out/f'{tag}_predictions.csv',index=False)
                rows.append({'dataset':dataset,'seed':seed,'condition':tag,'checkpoint':str(folder/'model.pt'),
                    'targets':len(frame),'mae':frame.ae.mean(),'mse':frame.se.mean(),
                    'delta_mae_vs_original':(pair.ae_condition-pair.ae_original).mean(),
                    'delta_mse_vs_original':(pair.se_condition-pair.se_original).mean()})
                print(f'{dataset} seed {seed} {tag} complete',flush=True)
            pd.DataFrame([r for r in rows if r['dataset']==dataset and r['seed']==seed]).to_csv(out/'metrics.csv',index=False)
            if not sharded:
                pd.DataFrame(rows).to_csv(output/'same_checkpoint_by_seed.csv',index=False)
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
    if sharded:
        return
    df=pd.DataFrame(rows)
    df.groupby(['dataset','condition'],as_index=False).agg(
        mae=('mae','mean'),mse=('mse','mean'),delta_mae=('delta_mae_vs_original','mean'),
        delta_mse=('delta_mse_vs_original','mean'),seeds=('seed','nunique')).to_csv(output/'same_checkpoint_summary.csv',index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['paired','inference','aggregate','all'],default='all')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output',type=Path,default=PROJECT/'revision'/'reviewer2'/'results')
    parser.add_argument('--bootstrap-reps',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=20261122)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS), help='Datasets for checkpoint inference.')
    parser.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS), help='Seeds for checkpoint inference.')
    parser.add_argument('--original-only',action='store_true',help='Validate checkpoint reproduction without interventions.')
    parser.add_argument('--sharded',action='store_true',help='Write only per-seed inference summaries for parallel workers.')
    parser.add_argument('--skip-mimic',action='store_true',help='Exclude MIMIC while Warpformer exports are still running.')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.mode in ('paired','all'): paired(args.output,args.bootstrap_reps,args.seed,args.skip_mimic)
    if args.mode in ('inference','all'): inference(args.output,args.device,args.datasets,args.seeds,args.original_only,args.sharded)
    if args.mode == 'aggregate':
        files=[args.output/'same_checkpoint'/dataset/f'seed{seed}'/'metrics.csv' for dataset in DATASETS for seed in SEEDS]
        missing=[str(p) for p in files if not p.is_file()]
        if missing: raise RuntimeError(f'Missing inference summaries: {missing}')
        frame=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
        if len(frame)!=len(DATASETS)*len(SEEDS)*len(CONDITIONS): raise RuntimeError('Incomplete inference conditions')
        frame.to_csv(args.output/'same_checkpoint_by_seed.csv',index=False)
        frame.groupby(['dataset','condition'],as_index=False).agg(
            mae=('mae','mean'),mse=('mse','mean'),delta_mae=('delta_mae_vs_original','mean'),
            delta_mse=('delta_mse_vs_original','mean'),seeds=('seed','nunique')).to_csv(args.output/'same_checkpoint_summary.csv',index=False)
    print('Finished',args.mode,'outputs:',args.output,flush=True)

if __name__=='__main__':main()
