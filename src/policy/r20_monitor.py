\
\
import os,re,sys,time
from pathlib import Path
ROOT=Path('.'); SC=Path('cache')
EPOCHS=8; FOLDS=4
G='\033[32m'; Y='\033[33m'; D='\033[2m'; B='\033[1m'; R='\033[0m'; CL='\033[2J\033[H'
def read(p):
    try: return Path(p).read_text()
    except Exception: return ''
def bar(frac,w=32):
    n=int(round(frac*w)); return '['+'#'*n+'-'*(w-n)+f'] {frac*100:5.1f}%'
def train_state(m):
    log=read(ROOT/f'artifacts/{m}/train.log')
    if 'done' in log.split('\n')[-2:][0] or re.search(r'^done',log,re.M): return 1.0,'trained + scored',None
    ep=re.findall(r'fold (\d) epoch (\d+) loss ([\d.]+) (\d+)s',log); sc=re.findall(r'^fold (\d) scored',log,re.M)
    if not ep and not sc:
        return (0.0,'waiting',None) if not log else (0.0,'loading tokens',None)
    f=int(ep[-1][0]) if ep else int(sc[-1][0]); e=int(ep[-1][1])+1 if ep else EPOCHS
    if sc and int(sc[-1][0])==f and (not ep or int(ep[-1][0])<=f): frac=(f+1)/FOLDS; st=f'fold {f} scored'
    else: frac=(f+ (e/EPOCHS)*0.45)/FOLDS + (0.0 if e<EPOCHS else 0.0); st=f'fold {f} epoch {e}/{EPOCHS} loss {ep[-1][2]}'
                                                          
    per=float(ep[-1][3])/e if ep else 22
    left=(FOLDS-f-1)*(EPOCHS*per+230)+ (max(0,EPOCHS-e)*per+230 if not (sc and int(sc[-1][0])==f) else 0)
    return frac,st,left
def exists(p): return Path(p).exists()
t0=time.time()
try:
    while True:
        lines=[f'{B}R20 build monitor{R}   {time.strftime("%H:%M:%S")}   watching {D}{time.time()-t0:5.0f}s{R}','']
        total_left=0
        for m,label in [('seq_v1s2','seed 2 training'),('seq_v1s3','seed 3 training')]:
            frac,st,left=train_state(m); col=G if frac>=1 else Y
            lines.append(f'{label:22s} {col}{bar(frac)}{R}  {st}' + (f'   ~{left/60:.0f} min left' if left else ''))
            if left: total_left=max(total_left,left)
        lines.append('')
        steps=[('seed 2 rescore full/eval',exists(ROOT/'artifacts/seq_v1s2/eval_all_lpo.csv')),
               ('seed 3 rescore full/eval',exists(ROOT/'artifacts/seq_v1s3/eval_all_lpo.csv')),
               ('seed runs complete',('SEEDS_DONE' in read(ROOT/'artifacts/seq_v1s3/dev_eval.txt'))),
               ('seed 2 window rescore',exists(ROOT/'artifacts/seq_v1s2/dev_last_2000_lpo.csv')),
               ('seed 3 window rescore',exists(ROOT/'artifacts/seq_v1s3/dev_last_2000_lpo.csv')),
               ('window fix-ups complete',('SEEDSFIX_DONE' in read(ROOT/'artifacts/seq_v1s3/dev_eval.txt'))),
               ('R20 blend + dev evaluation',('ens_gs21' in read(SC/'build_r20.log'))),
               ('R20 assembled',exists(ROOT/'artifacts/candidate_r20/submission.csv')),
               ('R20 validated',exists(ROOT/'artifacts/candidate_r20/validation.json')),
               ('copied to ~/Downloads/submission_r20.csv',exists(Path.home()/'Downloads/submission_r20.csv')),
               ('R20_DONE',('R20_DONE' in read(SC/'build_r20.log')))]
        done=sum(1 for _,ok in steps); ndone=sum(1 for _,ok in steps if ok)
        lines.append(f'{B}pipeline{R}  {bar(ndone/len(steps))}')
        for name,ok in steps: lines.append(f'  {G+"✔" if ok else D+"·"}{R} {name}')
        lines.append('')
        if total_left: lines.append(f'{D}estimated time to R20 file: ~{(total_left+8*60)/60:.0f} min (training ETA + rescoring/build){R}')
        log=read(SC/'build_r20.log').strip().split('\n')[-6:]
        if log and log!=['']: lines.append(f'{D}--- build_r20.log (tail) ---{R}'); lines+= [D+l[:110]+R for l in log]
        if 'R20_DONE' in read(SC/'build_r20.log'):
            lines.append(''); lines.append(f'{G}{B}R20 is ready: ~/Downloads/submission_r20.csv{R}')
        sys.stdout.write(CL+'\n'.join(lines)+'\n'); sys.stdout.flush()
        if 'R20_DONE' in read(SC/'build_r20.log'): break
        time.sleep(2)
except KeyboardInterrupt: pass
