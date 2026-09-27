import polars as pl

def window_actions(a, window):
    bounds={'first_2000':(0,2000),'last_2000':(1000,3000)}
    if window in bounds: lo,hi=bounds[window]
    else:
                                                                              
        lo,hi=map(int,window[1:].split('_'))
    C=pl.col
    a=a.filter((C('phase')=='development')&(C('time_index')>=lo)&(C('time_index')<hi))
    styles=[f'{p}_{k}' for p in ['style','local_style'] for k in range(4)]
    a=a.drop(styles)
    keys=['player_id','phase','street_no'];hk=keys+['hand_id','time_bin']
    counts=a.group_by(hk).agg(pl.len().alias('hand_n'),*[(C('action_class')==k).sum().alias(f'hand_{k}') for k in range(4)])
    glob=counts.group_by(keys).agg(C('hand_n').sum().alias('global_n'),*[C(f'hand_{k}').sum().alias(f'global_{k}') for k in range(4)])
    local=counts.group_by(keys+['time_bin']).agg(C('hand_n').sum().alias('local_n'),*[C(f'hand_{k}').sum().alias(f'local_{k}') for k in range(4)])
    counts=counts.join(glob,on=keys).join(local,on=keys+['time_bin'])
    counts=counts.with_columns(*[((C(f'global_{k}')-C(f'hand_{k}')+1)/(C('global_n')-C('hand_n')+4)).cast(pl.Float32).alias(f'style_{k}') for k in range(4)])
    counts=counts.with_columns(*[((C(f'local_{k}')-C(f'hand_{k}')+20*C(f'style_{k}'))/(C('local_n')-C('hand_n')+20)).cast(pl.Float32).alias(f'local_style_{k}') for k in range(4)])
    return a.join(counts.select(hk+styles),on=hk)
