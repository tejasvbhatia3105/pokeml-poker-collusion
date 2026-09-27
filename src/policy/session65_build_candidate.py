from pathlib import Path
from session46_build_candidate import main as build
from session65_list_pressure_inference import ROOT,load_models,events,score_pair,EXTRA_COLUMNS
def main():
 assert (ROOT/'verification.json').exists()
 build(out=Path('artifacts/candidate_r33'),inference_root=ROOT,model_loader=load_models,event_function=events,score_function=score_pair,extra_inference_columns=EXTRA_COLUMNS,method='session64 fixed equal pressure59 and grounded conditional-list62 scores, no family splice; seed then fourfold averaging; below.05 retainsR30/R32',local_map=.7973334826762246)
if __name__=='__main__':main()
