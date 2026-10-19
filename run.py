#!/usr/bin/env python3
import argparse,copy,json
from pathlib import Path

MODELS=['RF','XGB','SVM','BERT','DistilBERT','T5','DeBERTa','LiMP']

def training_options(p):
    p.add_argument('--data',required=True);p.add_argument('--out',default='results')
    p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS);p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--split-seed',type=int,default=42);p.add_argument('--val-fraction',type=float,default=.2)
    p.add_argument('--epochs',type=int,default=77);p.add_argument('--patience',type=int,default=10,help='0 runs all epochs')
    p.add_argument('--mode',choices=['frozen','full'],default='frozen',help='Frozen is a distinct low-memory protocol, not full fine-tuning')
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--accumulation',type=int,default=16)
    p.add_argument('--max-length',type=int,default=128);p.add_argument('--encoder-lr',type=float,default=2e-5);p.add_argument('--head-lr',type=float,default=1e-4)
    p.add_argument('--dropout',type=float,default=.2);p.add_argument('--weight-decay',type=float,default=.01);p.add_argument('--prototypes',type=int,default=3)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--class-weights',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--temperature',action='store_true')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--trees',type=int,default=300);p.add_argument('--max-features',type=int,default=20000);p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--debug-samples',type=int,default=0,help='Neural debugging subset only. Never use for paper results.')
    p.add_argument('--overwrite',action='store_true');p.add_argument('--keep-going',action='store_true')

def suite(args):
    from limp.train import run_models
    variants=[]
    if args.kind=='component':variants=[(s,dict(ablation=s)) for s in ['full','no_linguistic','no_group_gating','no_conditioning','no_residual','no_prototypes','no_refinement']]
    if args.kind=='linguistic':
        from limp.data import GROUPS
        variants=[('all_groups',{})]+[('without_'+g,dict(omit_group=i)) for i,g in enumerate(GROUPS)]
    if args.kind=='prototype':variants=[(f'K_{k}',dict(prototypes=k)) for k in [1,2,3,4,5]]
    if args.kind=='hyperparameter':
        variants=[('reference',{})]+[(f'{key}_{v}',{key:v}) for key,values in [('head_lr',[5e-5,2e-4]),('dropout',[.1,.3]),('max_length',[64,256]),('encoder_lr',[1e-5,3e-5])] for v in values if key!='encoder_lr' or args.mode=='full']
    root=Path(args.out)
    for label,changes in variants:
        a=copy.deepcopy(args);a.models=['LiMP'];a.out=str(root/'experiments'/args.kind/label)
        for k,v in changes.items():setattr(a,k,v)
        run_models(a)
    print('Run report on the parent results directory to collect suite tables.')

def main():
    p=argparse.ArgumentParser(description='Actual LiMP and baseline evaluation. No hard-coded results.');sub=p.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train');training_options(tr)
    su=sub.add_parser('suite');training_options(su);su.add_argument('--kind',choices=['component','linguistic','prototype','hyperparameter'],required=True)
    re=sub.add_parser('report');re.add_argument('--out',default='results');re.add_argument('--seed',type=int,default=42);re.add_argument('--bootstrap',type=int,default=2000)
    ex=sub.add_parser('explain');ex.add_argument('--out',default='results');ex.add_argument('--model',choices=MODELS,default='LiMP');ex.add_argument('--seed',type=int,default=42);ex.add_argument('--samples',type=int,default=3);ex.add_argument('--shap-evals',type=int,default=512);ex.add_argument('--lime-samples',type=int,default=1000);ex.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    to=sub.add_parser('topics');to.add_argument('--out',default='results');to.add_argument('--topics',type=int,default=6)
    st=sub.add_parser('independent-datasets');st.add_argument('--csv',required=True);st.add_argument('--out',default='independent_dataset_statistics');st.add_argument('--confirm-independent',action='store_true',required=True)
    pr=sub.add_parser('predict');pr.add_argument('--out',default='results');pr.add_argument('--model',choices=MODELS,default='LiMP');pr.add_argument('--seed',type=int,default=42);pr.add_argument('--text',required=True);pr.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args()
    if a.command in ['train','suite']:
        if a.epochs<1 or a.batch_size<1 or a.accumulation<1 or a.prototypes<1:raise ValueError('Positive epochs, batch, accumulation, prototypes required')
        if not 0<a.val_fraction<1:raise ValueError('val-fraction must be in (0,1)')
        if a.debug_samples and 'debug' not in a.out.lower():raise ValueError('Use an output path containing debug for debugging runs')
        if a.command=='train':
            from limp.train import run_models
            run_models(a)
        else:suite(a)
    elif a.command=='report':
        from limp.report import report
        print(report(a.out,a.seed,a.bootstrap))
    elif a.command=='explain':
        from limp.explain import explain
        explain(a.out,a.model,a.seed,a.samples,a.shap_evals,a.lime_samples,a.device)
        # Rebuild workbook including attribution-faithfulness table.
        from limp.report import report
        report(a.out,a.seed)
    elif a.command=='topics':
        from limp.topics import topics
        topics(a.out,a.topics)
    elif a.command=='independent-datasets':
        from limp.statistics import independent_dataset_tests
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True);pivot,ranks,cd=independent_dataset_tests(a.csv,out)
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4));order=ranks.argsort();ax.plot(ranks[order],range(len(order)),'o',color='#8c719b');ax.set_yticks(range(len(order)),pivot.columns[order]);ax.set(xlabel='Average rank (lower is better)',title=f'Independent-dataset ranks; Nemenyi CD={cd:.3f}');ax.errorbar(ranks[order],range(len(order)),xerr=cd/2,fmt='none',color='#9aa6b4');fig.tight_layout();fig.savefig(out/'critical_difference_rank_intervals.png',dpi=240)
    else:
        from limp.explain import Predictor
        from limp.data import LABELS
        z=Predictor(a.out,a.model,a.seed,a.device)([a.text])[0];print(json.dumps(dict(zip(LABELS,map(float,z))),indent=2))

if __name__=='__main__':main()
