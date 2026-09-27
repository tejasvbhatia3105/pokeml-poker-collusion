import os
import numpy as np, os
from pathlib import Path
S=Path(os.environ.get('POKEML_SCRATCH','cache'))
for sub,key in [('seq_tokens','X'),('action_tokens','XA')]:
    n=0
    for p in sorted((S/sub).glob('*.npz')):
        z=np.load(p)
        if key not in z.files: continue
        xp=S/sub/(p.stem+f'_{key}.npy')
        if not xp.exists() or os.path.getsize(xp)<1000: np.save(xp,z[key])
        keep={k:z[k] for k in z.files if k!=key}; tmp=p.with_suffix('.tmp.npz'); np.savez(tmp,**keep); os.replace(tmp,p); n+=1
    print(sub,'stripped',n,flush=True)
