\
\
\
\
\
\
\
import sys
from pathlib import Path
import session105_rollout_heads as heads
import session108_action_frontier as frontier

ROOT=Path('artifacts/evidence_session109_frontier_heads')

def configure():
    heads.ROOT=ROOT;heads.roll=frontier
    return heads

def main():
    h=configure();command=sys.argv[1]
    if command in ['baseline','prepare','train']:getattr(h,command)()
    elif command=='compare':
        import session105_compare as compare
        compare.ROOT=ROOT;original=compare.compare
        compare.compare=lambda path,columns,prefix:original(path,columns,'session109')
        compare.main()
    elif command=='verify':
        import session105_verify as verify
        verify.main()
    else:raise ValueError(command)

if __name__=='__main__':main()
