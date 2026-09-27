import sys
from pathlib import Path
import session105_rollout_heads as heads
import session112_integrated_values as values
ROOT=Path('artifacts/evidence_session113_precision_heads')

def main():
    heads.ROOT=ROOT;heads.roll=values;command=sys.argv[1]
    if command in ['baseline','prepare','train']:getattr(heads,command)()
    elif command=='compare':
        import session105_compare as compare
        compare.ROOT=ROOT;original=compare.compare
        compare.compare=lambda path,columns,prefix:original(path,columns,'session113')
        compare.main()
    elif command=='verify':
        import session105_verify as verify
        verify.main()
    else:raise ValueError(command)

if __name__=='__main__':main()
