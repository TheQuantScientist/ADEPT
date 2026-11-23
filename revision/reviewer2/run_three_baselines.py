#!/usr/bin/env python3
"""Paired ADEPT vs APN, GraFITi, and tPatchGNN on verified common targets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_reviewer2 as core

MODELS=('APN','GraFITi','tPatchGNN')
SOURCE=Path('/home/nckh2/qa/ChronoLM/vendor/upstream_benchmark/storage/results/cross_baseline')
SPECS={'P12':('P12','sl36_pl3'),'MIMIC':('MIMIC_III','sl72_pl3'),
       'USHCN':('USHCN','sl150_pl3'),'HumanActivity':('HumanActivity','sl3000_pl300')}


def locate(dataset,model):
    dirname,tag=SPECS[dataset]; root=SOURCE/dirname/model; found={}
    for path in sorted(root.rglob('output_pred.npy')):
        if tag not in str(path):continue
        d=path.parent
        if not all((d/n).is_file() for n in ('input_y.npy','input_y_mask.npy','input_sample_ID.npy','eval_configs.yaml')):continue
        cfg=yaml.safe_load((d/'eval_configs.yaml').read_text()) or {}
        if cfg.get('history_perturbation','original')!='original' or float(cfg.get('history_keep_fraction',1.0))!=1.0:continue
        it=next((x for x in path.parts if x.startswith('iter') and x[4:].isdigit()),None)
        if it is None:continue
        index=int(it[4:])
        if index in found:raise ValueError(f'Ambiguous {dataset}/{model}/{it}: {found[index]} and {d}')
        found[index]=d
    if sorted(found)!=[0,1,2]:raise FileNotFoundError(f'{dataset}/{model}: expected iterations 0,1,2; found {sorted(found)}')
    return found


def match(adept,baseline,dataset,model):
    keys=['Entity','Variable','Step']
    method='entity_variable_step'
    if dataset=='HumanActivity' and model=='tPatchGNN':
        # Padded time indices differ; align the full chronological sequence
        # within each entity and variable, then verify every actual value.
        adept=adept.sort_values(keys).copy();baseline=baseline.sort_values(keys).copy()
        adept['ordinal']=adept.groupby(['Entity','Variable']).cumcount()
        baseline['ordinal']=baseline.groupby(['Entity','Variable']).cumcount()
        keys=['Entity','Variable','ordinal'];method='within_variable_chronological_order'
    pair=adept.merge(baseline,on=keys,how='inner',validate='one_to_one',suffixes=('','_baseline'))
    if pair.empty:raise ValueError(f'{dataset}/{model}: no matched targets')
    if dataset=='HumanActivity' and model=='tPatchGNN' and (len(pair)!=len(adept) or len(pair)!=len(baseline)):
        raise ValueError('HumanActivity/tPatchGNN chronological mapping does not have complete coverage')
    if not np.allclose(pair.Actual_scaled,pair.baseline_actual,atol=2e-5,rtol=1e-4):
        bad=int((~np.isclose(pair.Actual_scaled,pair.baseline_actual,atol=2e-5,rtol=1e-4)).sum())
        raise ValueError(f'{dataset}/{model}: {bad} matched target values differ')
    # Keep ADEPT's original target coordinate even for ordinal alignment.
    return pair,method


def calculate(dataset,model,output,reps,rng,human_mapping):
    dirs=locate(dataset,model);names=core.variable_names(dataset);combined=None;cover=[];method=None
    for i,seed in enumerate(core.SEEDS):
        adept,apath=core.standard_frame(dataset,seed)
        baseline=core.baseline_frame(dirs[i],dataset,names,human_mapping if dataset=='HumanActivity' else None)
        pair,method=match(adept,baseline,dataset,model)
        keys=['Entity','Variable','Step']
        cols=pair[keys+['Actual_scaled']].copy()
        cols[f'adept_ae_{i}']=(pair.Actual_scaled-pair.Predicted_scaled).abs()
        cols[f'baseline_ae_{i}']=(pair.baseline_actual-pair.baseline_pred).abs()
        cover.append((len(adept),len(baseline),len(pair)))
        combined=cols if combined is None else combined.merge(cols,on=keys,suffixes=('',f'_{i}'),validate='one_to_one')
    if any(len(combined)!=x[2] for x in cover):raise ValueError(f'{dataset}/{model}: target support differs across iterations')
    for i in (1,2):
        if not np.allclose(combined.Actual_scaled,combined[f'Actual_scaled_{i}'],atol=2e-5,rtol=1e-4):
            raise ValueError(f'{dataset}/{model}: ADEPT targets differ across seeds')
    combined['adept_mean_ae']=combined[[f'adept_ae_{i}' for i in range(3)]].mean(axis=1)
    combined['baseline_mean_ae']=combined[[f'baseline_ae_{i}' for i in range(3)]].mean(axis=1)
    combined['delta_adept_minus_baseline']=combined.adept_mean_ae-combined.baseline_mean_ae
    combined['cluster']=combined.Entity.str.split('_').str[0] if dataset=='HumanActivity' else combined.Entity
    groups=combined.groupby('cluster').agg(delta=('delta_adept_minus_baseline','sum'),n=('Step','size')).to_numpy(float)
    n=len(groups);out=np.empty(reps)
    for j in range(reps):
        x=groups[rng.integers(0,n,size=n)].sum(axis=0);out[j]=x[0]/x[1]
    output.mkdir(parents=True,exist_ok=True)
    combined.to_csv(output/f'{dataset}_{model}_paired_targets.csv',index=False)
    return {'dataset':dataset,'baseline':model,'matching_method':method,'common_targets':len(combined),
      'adept_targets_per_seed':cover[0][0],'baseline_targets_per_seed':cover[0][1],
      'adept_coverage':len(combined)/cover[0][0],'clusters':n,
      'cluster_unit':'subject' if dataset=='HumanActivity' else 'entity',
      'adept_mae_common':combined.adept_mean_ae.mean(),'baseline_mae_common':combined.baseline_mean_ae.mean(),
      'delta_adept_minus_baseline':combined.delta_adept_minus_baseline.mean(),
      'ci_low':np.quantile(out,.025),'ci_high':np.quantile(out,.975),
      'baseline_directories':';'.join(str(dirs[i]) for i in range(3)),
      'adept_source':'saved standard-window ADEPT prediction logs',
      'note':'subject bootstrap has only 3 clusters; interpret cautiously' if dataset=='HumanActivity' else ''}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'three_baselines_results')
    parser.add_argument('--bootstrap-reps',type=int,default=5000);parser.add_argument('--seed',type=int,default=20261122)
    args=parser.parse_args()
    if args.bootstrap_reps<1:parser.error('--bootstrap-reps must be positive')
    args.output.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(args.seed)
    human_mapping=core.human_map();rows=[];fail=[]
    for dataset in core.DATASETS:
        for model in MODELS:
            try:
                row=calculate(dataset,model,args.output,args.bootstrap_reps,rng,human_mapping);rows.append(row)
                print(f'{dataset}/{model}: matched {row["common_targets"]}/{row["adept_targets_per_seed"]}, MAE delta {row["delta_adept_minus_baseline"]:.6f}',flush=True)
            except Exception as exc:
                fail.append({'dataset':dataset,'model':model,'error':str(exc)})
                print(f'FAILED {dataset}/{model}: {exc}',file=sys.stderr,flush=True)
            pd.DataFrame(rows).to_csv(args.output/'paired_three_baselines.csv',index=False)
            (args.output/'failures.json').write_text(json.dumps(fail,indent=2))
    print(f'Completed {len(rows)}/12 comparisons. Output: {args.output}',flush=True)
    if fail:sys.exit(1)

if __name__=='__main__':main()
