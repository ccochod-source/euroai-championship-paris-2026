# Clément (+ Devin) — Unlearning status, 21 Nov 21:00

Complement to BOARD.md / RESULTS.csv / SUBMISSIONS.md (updated 21:05).
To be copied into BOARD.md / RESULTS.csv / SUBMISSIONS.md when they exist.

## Access
- The cochod2 account has NO GPU budget left on JURECA since 20:38 (JSC email). No job possible under cochod2.
- Files and login still accessible.

## Official submissions (public score)
| Time | Candidate | Score |
|---|---|---|
| - | original model | 0.0 |
| - | corrected head x1.0 / x1.07 / x1.10 | 0.0082 / 0.0099 / 0.0101 (best) |
| - | global temperature T=3.789 | 0.0053 |
| - | second head x1.035 | 0.0059 |
| 18:49 | per-class correction | 0.0084 |
| 20:47 | last 6 blocks distilled towards an out-of-fold DINOv2 teacher (oof-d6-lr3e-5-s300) | 0.0076 |

Next submission possible regarding the delay: right now (last submission 20:47). No submission planned without approval.

## What this suggests (3 points, not a certainty)
- Mean loss on forget: best = 1.58 -> 0.0101; candidates with loss 0.33 -> 0.0053 and 0.0076.
- The gold models seem much less confident on forget than the DINOv2 teacher (loss 0.44, 85.7% agreement with the original ViT).
- Tiny ImageNet is a poor retention control (competition images = generated images).

## Tracks already done (do not redo)
- Head corrections (scale, per class, SALSA, margins, ridge, public anchoring): ceiling ~0.010.
- JiT, LF-SSD: rejected locally.
- Last-block MLP on cached activations: marginal local gain.
- Block distillation towards an out-of-fold teacher, tau=1: 0.0076 official.

## Track I am taking
- Same block distillation with a softened teacher (tau = 3, 5, 8 -> target loss 1.8 to 3.2).
- Ready to launch, but it needs a teammate's GPU:
  CUDA_VISIBLE_DEVICES=<n> bash /p/scratch/training2666/cochod2/euroai-paris/unlearning/oof/run-sweep.sh /p/scratch/training2666/cochod2/euroai-paris/unlearning/output/bruno 6:3e-5:3 6:3e-5:5 12:3e-5:3
  (format depth:lr:tau, ~3 min per setting, writes logs + patches in output/bruno, submits nothing)

## Files
- Scripts: unlearning/oof/ (oof_finetune.py, run-sweep.sh, teacher oof-wd0.01.pt)
- Results of job 15902873: unlearning/output/oof-15902873/ (logs + patches per step)
