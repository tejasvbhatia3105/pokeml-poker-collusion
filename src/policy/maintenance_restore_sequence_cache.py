import argparse,hashlib,json,os,shutil,zipfile
from pathlib import Path
from maintenance_repack_sequence_cache import MANIFEST,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['seq_tokens','action_tokens']);p.add_argument('--table');p.add_argument('--all',action='store_true');a=p.parse_args()
    if not a.all and not (a.kind and a.table):p.error('Use --kind and --table for one cache, or --all.')
    records={r['unpacked']:r for r in map(json.loads,MANIFEST.read_text().splitlines()) if r['status']=='complete'}
    selected=[r for name,r in records.items() if (a.all or (Path(name).parent.name==a.kind and Path(r['archive']).stem==a.table))]
    assert selected,'No verified cache matches this selection.'
    pending=[];required=0
    for r in selected:
        path=Path(r['unpacked']);member='XA.npy' if path.name.endswith('_XA.npy') else 'X.npy';expected=r['verified_uncompressed_member_sha256'][member]
        if path.exists():
            with path.open('rb') as f:assert digest(f)==expected
            continue
        with zipfile.ZipFile(r['archive']) as z:required+=z.getinfo(member).file_size
        pending.append((r,path,member,expected))
    assert shutil.disk_usage('.').free>required+1024**3,'Insufficient space with1GiB reserve; restore fewer tables.'
    for r,path,member,expected in pending:
        temporary=path.with_suffix('.restore.npy');assert not temporary.exists()
        with zipfile.ZipFile(r['archive']) as z,z.open(member) as src,temporary.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
        with temporary.open('rb') as f:assert digest(f)==expected
        os.replace(temporary,path)
    print('RESTORED',len(pending),'BYTES',required)

if __name__=='__main__':main()
