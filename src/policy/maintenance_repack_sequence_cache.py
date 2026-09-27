\
\
\
\
\
\
from pathlib import Path
import copy,hashlib,json,os,shutil,sys,zipfile

BASE=Path('cache')
MANIFEST=Path('artifacts/sequence_cache_repack.jsonl')

def digest(stream):
    h=hashlib.sha256()
    while chunk:=stream.read(1024*1024):h.update(chunk)
    return h.hexdigest()

def main(limit=None):
    done=0;freed=0
    for dirname,suffix,member in [('seq_tokens','_X.npy','X.npy'),('action_tokens','_XA.npy','XA.npy')]:
        for unpacked in sorted((BASE/dirname).glob('*'+suffix)):
            archive=unpacked.with_name(unpacked.name[:-len(suffix)]+'.npz')
            if not archive.exists():continue
            temporary=archive.with_suffix('.repack.npz');assert not temporary.exists(),temporary
            before=archive.stat().st_size+unpacked.stat().st_size
            with unpacked.open('rb') as f:array_hash=digest(f)
            with archive.open('rb') as f:old_archive_hash=digest(f)
            hashes={}
            with zipfile.ZipFile(archive) as src:
                if member in src.namelist():
                    with src.open(member) as f:
                        if digest(f)!=array_hash:print('SKIP_DISTINCT_ARRAYS',unpacked,flush=True);continue
                with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as dst:
                    for info in src.infolist():
                        with src.open(info) as f:hashes[info.filename]=digest(f)
                        cloned=copy.copy(info);cloned.compress_type=zipfile.ZIP_DEFLATED;cloned._compresslevel=1
                        with src.open(info) as a,dst.open(cloned,'w',force_zip64=True) as b:shutil.copyfileobj(a,b,1024*1024)
                    if member not in hashes:dst.write(unpacked,arcname=member,compress_type=zipfile.ZIP_DEFLATED,compresslevel=1)
                    hashes[member]=array_hash
            with zipfile.ZipFile(temporary) as check:
                assert set(check.namelist())==set(hashes)
                for name,expected in hashes.items():
                    with check.open(name) as f:assert digest(f)==expected,(archive,name)
            with temporary.open('rb') as f:new_archive_hash=digest(f)
            after=temporary.stat().st_size
            record=dict(archive=str(archive),unpacked=str(unpacked),old_archive_sha256=old_archive_hash,new_archive_sha256=new_archive_hash,
                verified_uncompressed_member_sha256=hashes,before_bytes=before,after_bytes=after,freed_bytes=before-after)
            with MANIFEST.open('a') as log:log.write(json.dumps(dict(status='verified_before_replace',**record))+'\n');log.flush();os.fsync(log.fileno())
            os.replace(temporary,archive);unpacked.unlink()
            with MANIFEST.open('a') as log:log.write(json.dumps(dict(status='complete',**record))+'\n')
            done+=1;freed+=before-after
            if done%40==0 or limit:print('REPACKED',done,'FREED',freed,flush=True)
            if limit and done>=limit:return
    print('COMPLETE',done,'FREED',freed,flush=True)

if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else None)
