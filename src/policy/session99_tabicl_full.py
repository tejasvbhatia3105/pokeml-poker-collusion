\
\
\
\
\
from pathlib import Path
import session98_tabicl_events as s

ROOT = Path('artifacts/evidence_session99_tabicl_full')

def main():
    ROOT.mkdir(exist_ok=True)
    s.ROOT = ROOT
    s.CONFIG = dict(s.CONFIG, method=__doc__, features=929,
                    feature_selection='All929 original inputs, original order; no selection',
                    checkpoint=str(s.CHECKPOINT))
    s.main()

if __name__=='__main__':
    main()
