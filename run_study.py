"""Sequential orchestration. Each model finishes before the next starts."""
import argparse,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/Claim_Credibility.xlsx');p.add_argument('--out',default='results');p.add_argument('--mode',choices=['frozen','full'],default='frozen');p.add_argument('--seeds',nargs='+',default=['42']);p.add_argument('--epochs',default='77');p.add_argument('--patience',default='10');p.add_argument('--suites',action='store_true');p.add_argument('--explanations',action='store_true');p.add_argument('--topics',action='store_true');a=p.parse_args()
    script=str(Path(__file__).resolve().with_name('run.py'))
    def run(*x):subprocess.run([sys.executable,script,*map(str,x)],check=True)
    common=['--data',a.data,'--out',a.out,'--mode',a.mode,'--epochs',a.epochs,'--patience',a.patience,'--seeds',*a.seeds]
    run('train',*common)
    if a.suites:
        for kind in ['component','linguistic','prototype','hyperparameter']:run('suite',*common,'--kind',kind)
    run('report','--out',a.out,'--seed',a.seeds[0])
    if a.explanations:run('explain','--out',a.out,'--seed',a.seeds[0],'--samples',3)
    if a.topics:run('topics','--out',a.out)

if __name__=='__main__':main()
