\
\
\
\
\
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('artifacts/storage_20260915')

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ROOT.mkdir(exist_ok=True);paths=[]
    for p in Path('artifacts').rglob('*'):
        if p.suffix not in ['.csv','.npy'] or p.is_symlink() or any(x.startswith(('candidate','diagnostic')) for x in p.parts):continue
        st=p.stat()
        if st.st_size>5_000_000 and st.st_nlink==1 and st.st_blocks*512>.8*st.st_size:paths.append(p)
    paths.sort(key=lambda p:p.stat().st_size,reverse=True);rows=[];start=time.time()
    for i,p in enumerate(paths):
        temp=p.with_name(p.name+'.fscompression.tmp');assert not temp.exists()
        before=p.stat().st_blocks*512;logical=p.stat().st_size;sha=digest(p)
        result=subprocess.run(['/usr/bin/ditto','--hfsCompression','--noclone',str(p),str(temp)],capture_output=True,text=True)
        if result.returncode:
            if temp.exists():temp.unlink()
            raise RuntimeError(result.stderr)
        assert temp.stat().st_size==logical and digest(temp)==sha
        after=temp.stat().st_blocks*512
        if after<before:os.replace(temp,p)
        else:temp.unlink();after=before
        assert digest(p)==sha
        rows.append(dict(path=str(p),logical_bytes=logical,allocated_before=before,allocated_after=after,sha256=sha))
        (ROOT/'compression.json').write_text(json.dumps(dict(method=__doc__,files=rows,bytes_saved=sum(r['allocated_before']-r['allocated_after'] for r in rows)),indent=2))
        if i%20==0:print('compressed',i+1,len(paths),'saved_MB',sum(r['allocated_before']-r['allocated_after'] for r in rows)/1e6,'seconds',time.time()-start,flush=True)
    print('complete',len(rows),'saved_bytes',sum(r['allocated_before']-r['allocated_after'] for r in rows),flush=True)

if __name__=='__main__':main()
