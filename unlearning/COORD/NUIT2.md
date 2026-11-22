# NIGHT2 — Orehn's final instructions (01:40), valid until he wakes up
Single goal: RAISE THE SCORES in a robust way (final ranking = 70% hidden), fully autonomously within the rules below.

## Unlearning (priority 1)
1. ABSOLUTELY follow Clément's plan (BOARD.md / DEMANDES.md / SUBMISSIONS.md). His requests come first.
2. When his plan is exhausted or he has nothing queued: keep improving on your own (UL_STRATEGIE.md, your ft6 tracks, out-of-fold teachers).
3. Questions: if you need context, write a line "QUESTION -> Clément/Devin session: ..." in BOARD.md and re-read his answers at each 5 min loop.

## Model Equality (priority 2, simulator session)
ROBUSTESSE_70 audit then continuous improvement, without spending the reserve or submitting (the reserve requires Orehn's typed confirmation).

## Learning loop (both tasks)
- After EVERY result (local or official): 3-5 lines of post-mortem in the task's log: what changed, why (hypothesis), whether the prediction was right, next test.
- Every ~2 h: launch a "critic" sub-agent that re-reads the results without having produced the work and looks for errors, leaks, overfitting to the public split, unproven conclusions. Integrate its remarks.
- Do targeted literature/code search when a track is stuck.
- Anti-overfit: decide on local metrics + bootstrap; public gap < 0.01 = noise; when in doubt "different" (ME); local accuracy >= 75% (UL).

## Unchanged prohibitions
No 2nd GPU job, no sbatch/scancel on other people's jobs, no token displayed/copied, no spending of the ME reserve, no deleting other people's files or submitted checkpoints, SUBMIT_LOCK + 10 min.

## 07:00
Clear summary at the top of results/status.md (ME) and results/unlearning_status.md (UL): scores, what worked/failed, decisions for Orehn to take.
