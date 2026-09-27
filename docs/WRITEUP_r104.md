# 1st place solution

## Overview

My selected submissions are r104 and r103. They differ in one parameter of the 4th family evidence rule described below. R104 is the reproduced file. Code and setup instructions are here: https://github.com/tejasvbhatia3105/pokeml-poker-collusion. The approach infers coordination from betting, cards and chip flow only. Identifiers just join tables and define the hold-out pools.

## Pair Ranking

Each pair is described by the two players' actions and how they deviate from the ordinary policy model. The probability of every observed action is derived using CatBoost action and bet-size models, which are cross-fitted over 4 folds of whole player pools. The deviations were aggregated per pair into different channels like partner-facing calls, surrenders, hidden-card aggression, and passiveness in heads-up, with account for chip flow, showdown results and preflop equity.

A CatBoost pair model and transformers over per-hand token sequences (hand summaries and raw actions, three seeds each) share one positive-unlabelled scheme: the 372 labelled positives and 1,488 negatives, high-confidence pseudo-positives from unlabelled pairs, hard negatives formed by pairing a known colluder with a non-partner, and a large low-weight background sample. Because evaluation covers 2,000 hands against 3,000 in development, each labelled pair also contributes its first and last 2,000-hand crops. The final risk is the geometric mean of the two rankers. It is then recomputed leaving out hands in which a stronger partner of either player was dealt, so one colluder's deviation does not leak into their other pairs.

## Evidence for Disclosed Families

Development evidence lists were consistent with the earliest hands of a primary event type, followed by a fallback type, cut at 5. Per-family boosted rankers score every shared hand from policy deviation, relationship context, all the players' cards and the run-out equity. An ordered-list likelihood turns scores into lists of that shape, which are refined by an outward pressure model for isolation as well as a tree correction. Development MAP@5 reached 0.797 here, and further experiments moved nothing, so we looked elsewhere.

## The 4th Family

An `other_coordination` behavior was named with no development example. Had it existed in evaluation, its pairs would have been labelled with the closest disclosed family and given that family's evidence. So I looked for behavior that requires knowing a partner's cards. For every ordered pair (P, Q), over evaluation hands in which Q acts after P, I tried correlating P's opening aggression (P's first voluntary preflop action is a raise), residualized on P's own preflop equity, with Q's preflop equity.

Known directed-transfer pairs showed negative dependence in both phases, which calibrated the statistic. In the evaluation pairs the model labeled coordinated isolation, 20% showed strong positive dependence, vs 1% of development isolation pairs and a 0.5% null rate. The raiser opened with average equity of 0.52, raised 51% of the time when the partner was strong vs 17% otherwise, pots tripled, and chips ended up with the partner. No development positives behaved this way. Sweeping the lower risk areas brought the set to 77 pairs.

The metric scores an `other_coordination` pair as a negative in all 3 family rankings, so a high-risk hidden-family pair carrying a known family label is going to be a false positive in that family. So we label that group `other_coordination`, except the 16 with family probability of at least 0.98 and the weakest card dependence, which behave like ordinary isolation or directed-transfer pairs and keep their label and evidence.

## Evidence of the 4th Family

Those lists followed the same earliest-first rule observed in the disclosed families. Candidates were hands where P opens with a raise while Q's preflop equity was over 0.5 and P's at most 0.7. The eight best by the isolation ranker's hand score were kept and the 3 earliest listed. Then the reverse, Q opening when P is strong: the 5 best by the same score (8 in r103), the 2 earliest. All 5 were ordered by time.

The split follows the probe: dependence was strongest in one direction but present in both for about a third of the pairs, and same-direction lists beyond 3 hands added nothing. Other variants of this rule were tested and set aside, and the repo records them.

## Evidence as Ranking Signal

The mean of a pair's 5 best evidence hand scores separates confirmed positives from the 0.05 to 0.2 risk area with AUC 0.976, and shares no aggregate features with the rankers. Pairs that scored above 0.62 received a risk between 0.90 and 0.95, and pairs between 0.55 and 0.62 got between 0.86 and 0.90. 4th family pairs are at 0.995, or between 0.80 and 0.89 for lower-confidence sweeps, where lifted blocks were ordered by estimated purity. Weak-evidence pairs weren't demoted, as there are some genuine positives with unremarkable hands.

## Validation and Limits

Development checks held out whole player pools and measured recall at fixed false-positive budgets over all pairs, as labeled-only AP was already at 0.9999. The 4th family had no development tables: its detection rested on the probe's calibration against the disclosed families and the null rate, and its list rule on the same earliest-first structure.

## Reproduction

The repository rebuilds the policy caches from the 8 competition files, retrains the pair and sequence models (the transformers ran on a single rented GPU, everything else on a 16 GB laptop), runs the evidence stack, applies the probe, and lifts labels and list rules. The README gives the commands stage by stage. Dependencies are pinned (Python 3.14, CatBoost 1.2.10, polars 1.44, PyTorch 2.8) and `models/` holds the archived weights. Runtimes: caches and rankers 30 minutes on the laptop, each transformer 15 to 60 minutes on one GPU, final layers 4 minutes.

Verified after the competition: rerunning the ranking stages from raw data reproduces the archived ranking at correlation 0.9991 (496 of the top 500 shared), and the final-layer chain of 25 scripts agrees with the uploaded r104 on 112,523 of 112,540 rows. The 17 differing rows trace to per-table inference files that were unreadable on my machine when the uploaded file was built and that the scripted run now uses. Retraining the seven transformers on a rented GPU and re-blending gives correlation 0.9985 with the archived ranking, again sharing 494 of the top 500. The evidence stack was not retrained after the competition.

## Five Case Reviews

Hands are from the selected file, both players dealt, cards from the released logs. P is the player whose raises track the partner's cards.

**P0B07484C3B53** (HED9734DF275233, H722EF5D9D6D6E9, HB065185142C973). P three-bets 2-9 offsuit over a stranger's open while the partner holds K-9, opens 6-9 offsuit into the partner's pocket nines and folds to the partner's flop bet. P opens 5-3 offsuit while the partner holds K-7 and folds the turn. Suspicious: weak opens timed to the partner's strong hands, then surrender. Benign: a loose opener who folds to any resistance.

**P74B2F7B56792** (H4A5294AF3D17ED, H00F422EF97E61F, H58B8A59F05C5A1). P raises K-9 offsuit over an open while the partner holds Q-6 suited, calls the partner's re-raise, then folds. P opens A-Q into the partner's A-Q suited and folds to a re-raise. P raises K-9 suited, the partner re-raises J-T, P folds the flop. Suspicious: the partner's re-raises repeatedly squeeze out a third player after P's raise, and P concedes. Benign: a light three-bettor folding to four-bets from a partner who happens to be strong.

**P97D394207E30** (HFBBE914D3D381F, HF59594DAE2DB05, H5D21891A69F37B). P opens Q-9 and four-bets into the partner's K-J suited, then folds to a shove. P opens 7-4 offsuit while the partner holds A-T suited, the partner folds, and P loses 344 chips to a third player after raising the flop and turn. P three-bets A-8 over the partner's Q-9 suited open, calls a four-bet, folds the flop. Suspicious: repeated raising wars between partners that end in P's fold, P's weak raises tracking the partner's strong holdings. Benign: two aggressive neighbours folding correctly to big re-raises.

**PD07FAE981F21** (HC51AADEA8CFF1F, H98827F27E250F0, HEF6D1AB5F2740E). P three-bets Q-J suited over the partner's J-9 raise and wins 265 when the partner bets and then calls a river raise. P opens 9-4 suited, the partner calls with Q-T and folds to a flop bet. P raises pocket fours after the partner limps Q-J suited, and the partner folds. Suspicious: here chips flow to the raiser, and the partner's calls and folds support the raise rather than contest it. Benign: real-hand aggression, ordinary folds by the partner.

**P8A6E08695A3F** (HD0761B73A481FD, H9DE93526EA3F87, H851A8AC50F61DC). P three-bets T-4 suited over a stranger's open while the partner holds T-9, the partner folds, and P loses 319 in an 898-chip pot. P opens A-J, the partner three-bets K-8 offsuit, P calls and folds to the partner's flop bet. P opens 3-2 offsuit while the partner holds K-J, the partner folds, P wins 65 at showdown. Suspicious: light raises by both partners inflate pots collected by the partner or a third player. Benign: both three-bet wide, and the K-8 three-bet is an ordinary bluff.
