from pathlib import Path
from session46_build_candidate import main as build
from session51_matchup_inference import ROOT,KIND,load_models,events
def main():
 assert KIND=='current'
 assert Path('artifacts/evidence_session50_matchup/current/verification.json').exists()
 assert Path('artifacts/evidence_session50_matchup/representation_audit.json').exists()
 build(out=Path('artifacts/candidate_r32'),inference_root=ROOT,model_loader=load_models,event_function=events,method='session50 current-only made-hand comparison added to R31 paired fold/bet event heads; Cat-and-joint prior with original R30 correction inputs; seed then fold averaging',local_map=.7926471027479092)
if __name__=='__main__':main()
