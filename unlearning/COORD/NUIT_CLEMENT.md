# Unlearning night plan — Clément (Devin session) -> Orehn's agent
Version 3, 22 Nov ~01:50. Replaces versions 1 and 2. Clément says Orehn asked his agent to follow this plan. PROTOCOLE.md prevails in case of conflict. My session has no access to JURECA until Clément wakes up: I launch, read and submit nothing.

## Findings (official scores, public split)
| checkpoint | setting | std clean / crop | retain_est | score |
|---|---|---|---|---|
| clem_A03r2_g4 | train all, rpow 2 | 2.59 | 0.757 | 0.1631 |
| clem_A03r3_g4 | train all, rpow 3 | 3.26 | 0.750 | 0.2019 |
| clem_A03r4_g4 | train all, rpow 4 | 3.66 / 3.11 | 0.750 | 0.2091 |
| ft6_A03r5_g4_last6_t1 | last6, rpow 5, t1 1.0 | 3.76 / 3.34 | 0.752 | **0.2101** (best) |
(all: A=0.3, alpha 4, gamma 4, lr 1e-4, 3000 steps)
The "make_targets formula + rpow" family PLATEAUS around 0.21: +0.007 then +0.001, whether we train everything or last6, with or without t1, and even when the crop std goes up. Do not spend any more GPU time or submissions on rpow, k of last<k>, t1, gamma or lr within this family.
Already discarded (not submitted): lr > 1e-4 (retain_est < 0.70); A=0.15 rpow 2 (0.741); gamma 6/8 + t1 (0.740/0.736).

## Step 1 — per-image out-of-fold targets (TOP PRIORITY, goes ahead of everything in gpu3)
Finish the extraction + the 15 OOF teachers (Codex), produce the per-image target file.
First run = ONE SINGLE change from a scored point: the setting of clem_A03r4_g4 (train all, gamma 4, lr 1e-4, 3000 steps) with --targets <OOF file>. No last3, no other loss at the same time, otherwise we will not know what made the difference.
Check before training: on the targets themselves, record the accuracy vs label, the mean CE and its std (compare with 0.31 / 4.10 / 3.66 of the best model). Write these three numbers in BOARD.md.

## Step 2 — bisection on the strength of the OOF targets
Three strengths (the 1 / 0.7 / 1.4 planned by Codex). Keep those with retain_est >= 0.75, submit the best one, then halve the interval between the two best scores. One variable only.

## Step 3 — shape control (already queued, gpu0)
c3_clement_A045r4_g4 (A 0.45, rpow 4, train all): same target spread but more images correctly classified. Local reading only; submit it only if a submission is left and step 1 gave nothing. If it scored >= 0.21, it is the shape and not the strength of forgetting that matters: then aim for softer OOF targets in step 2.

## Mandatory conditions for any submission
- retain_est >= 0.75;
- std of tr/crop and tr/noise.2 >= 85% of the clean std;
- validation against model.pt (keys, shapes, float32, finite, < 450 MB);
- SUBMIT_LOCK, 10 min interval, a line in SUBMISSIONS.md and RESULTS.csv;
- AT MOST 3 submissions tonight, each decided by Orehn / his agent. A candidate from the rpow family is no longer worth a submission.

## Robustness
- The limit on the NUMBER of files on scratch has been reached (organisers' announcement 01:33): a write may fail. If a run crashes when saving: relaunch once, then move on. Do not create new venvs or caches made of thousands of small files; prefer a single .pt / .npz.
- Do not delete: orehn/ckpt/ft6_A03r5_g4_last6_t1.pt (best), clem_A03r4_g4.pt (train-all fallback). Copies of both on Clément's Mac and in my VM. Can be deleted for the quota: clem_A03r2_g4.pt, clem_A03r3_g4.pt, clem_A005_g2.pt, clem_A03r4_g4_last6.pt, and the checkpoints of my runs c2 / e2 once read.

## My queue
Removed: e2_clement_A03r6_g4_last6 (plateaued family). Running on gpu0: c2 (rpow 5 train all, read only), then c3. Delete them if Model Equality needs GPU 0.
