# European AI Championship Paris 2026 — Team Eugenia

Hackathon organised by CISPA and Fraunhofer AISEC at Epitech Paris, 21–22 November 2026. This repository contains
team Eugenia's work on the three tasks: code, launch scripts, plans, experiment logs and official scores.

Last public leaderboard we read (22 Nov, 10:33, before our last submissions):

| Task | Public score | Folder |
|---|---:|---|
| Watermark Localisation | 0.76 | [`watermark/`](watermark/) |
| Model Equality | 1.00 | [`model_equality/`](model_equality/) |
| Unlearning | 0.26 (final best 0.2642) | [`unlearning/`](unlearning/) |
| Overall | 0.78, rank 10 of 44 | |

These scores are computed on the public split. The final ranking uses a held-out split and was not published when
this repository was created.

## What is not in the repository

- the competition data (images, `train.zip` / `test.zip`, `forget_images.npz`);
- model weights (`model.pt`, 340 MB checkpoints, watermark models);
- Python environments and caches;
- the team API key: submission scripts contain `YOUR_TEAM_API_KEY`.

The scripts therefore do not run as is without the files provided by the organisers.

## Task 1 — Unlearning (`31-unlearning`)

### Solution

Full-model fine-tuning of the ViT-B/16 towards **per-image out-of-fold targets**, with a retain regulariser.

1. **Targets.** For each forget image we want the output distribution of a model that never saw it. We train
   teachers on folds of the forget set (DINOv2-based and a related architecture) and predict each image only with
   teachers that did not train on it. This gives `q_oof_clean`, one distribution per image. We sharpen it with a
   power `k` and renormalise per image (`q_oof_k275.pt` is `k = 2.75`).
2. **Forget loss.** KL divergence between the target and the model output on strongly augmented views of the forget
   images.
3. **Retain loss.** KL divergence to the original model on AI-generated control images, weighted by `gamma`.
4. **Trainable part.** The last 6 transformer blocks (`--train last6`), 3000 steps, learning rate 1e-4.

```bash
python -u orehn/src/finetune.py --name clem_oof_k275_g3_last6 \
    --targets orehn/cache/q_oof_k275.pt --gamma 3 --train last6 --lr 1e-4 --steps 3000
```

Every candidate was checked before submission: retain proxy at least 0.75, loss spread kept on cropped and noisy
copies, state dict validated against `model.pt` (keys, shapes, float32, finite).

### Official scores

| Step | Method | Score |
|---|---|---:|
| 21 Nov evening | Global temperature, per-class correction, head editing | 0.005 to 0.010 |
| 21 Nov 22:00 | Full fine-tuning towards "never seen" targets + KL to the original model on control images | 0.057 → 0.110 |
| 21 Nov night | More dispersed targets (`rpow` 2, 3, 4) | 0.163 → 0.202 → 0.209 |
| 22 Nov 01:39 | Same, last 6 blocks only | 0.2101 |
| 22 Nov 02:21 | **Per-image out-of-fold targets** | 0.2561 |
| 22 Nov 11:26 | Retain weight `gamma` 4 → 3 | 0.2599 |
| 22 Nov 11:57 | Target power 3 → 2.75 | **0.2642** |

### What worked, what did not

- **Worked:** replacing one target formula shared by all images with per-image out-of-fold targets. This is most of
  the gain (0.21 → 0.26).
- **Did not work:** editing only the head or a temperature; stronger forgetting (power 4: 0.2255); more retain weight
  and longer training (0.2514); additionally lowering the true class at `gamma` 3 (0.2471); mixing two teacher
  families (0.2366).
- **Lesson:** our local metrics (loss, spread, retain proxy) did not predict the official score. One extrapolation
  announced 0.27 to 0.30 for a submission that scored 0.2347.

Sweep around the best point, one parameter at a time (last 6 blocks, 3000 steps):

| Target power `k` (gamma 4) | Score | | Retain weight `gamma` (k = 3) | Score |
|---|---:|---|---|---:|
| 2.5 | 0.2535 | | 8 (4000 steps) | 0.2514 |
| 3 | 0.2561 | | 4 | 0.2561 |
| 4 | 0.2255 | | 3 | 0.2599 |
| | | | 2.5 | 0.2575 |

### Where to look

- `unlearning/orehn/src/`: main pipeline (`finetune.py`, `teacher.py`, `oof_vit.py`, `common.py`).
- `unlearning/orehn/logs/`: one JSON of metrics per run.
- `unlearning/COORD/`: three-person coordination (`PROTOCOLE.md`, `BOARD.md`), every submission with its score
  (`SUBMISSIONS.md`), every run (`RESULTS.csv`), night plans.
- `unlearning/codex/`, `unlearning/oof/`, `unlearning/eugenia-trials/`, `unlearning/devin-vm/`: approaches explored
  before the final pipeline (heads, temperatures, multi-view distillation, first out-of-fold teachers).

## Task 2 — Watermark Localisation

### Solution

Two complementary approaches, described in `watermark/bruno-wm/s3/README.md` and `watermark/orehn/STATUS.md`:

- **Training-free rectangle search** ("propose and verify"): search for the crop that maximises decoding of the known
  message with the official decoder.
- **Segmentation networks** fine-tuned per watermark method (PixelSeal, TrustMark, StegaStamp) on the real training
  images, with per-method thresholds.

Key finding recorded by the team: for TrustMark and PixelSeal, the watermark in the challenge images is not
pixel-wise correlated with what the provided embedder produces, so networks trained on our own watermarked images
fail on the real ones. Everything was validated on the real training images.

Best public score recorded in `watermark/orehn/STATUS.md`: 0.7589 (Bruno's file). An ensemble estimated at 0.7619
locally scored 0.7448 on the public split.

### Where to look

`watermark/bruno/` and `watermark/bruno-wm/` (experiments, rectangle search, segmentation), `watermark/orehn/`
(ensembles), `watermark/wmloc/` and `watermark/README.txt` (organisers' starter kit).

## Task 3 — Model Equality (`33-model-equality`)

### Solution

A local simulator (`model_equality/sim/`) reproduces the problem with open models (Llama-3.2-1B, Qwen2.5-1.5B/3B,
SmolLM2-360M), their quantised, LoRA and distilled variants, and disguising system prompts. It was used to design
and validate the query protocol and the decision rule before spending the real query budget (100 answers per
endpoint). Public score: 1.00.

### Where to look

`model_equality/sim/` (dataset generation `make_spec*.py`, likelihoods `lik*.py`, pair classifier `pair_clf.py`,
validation `validate_*.py`), `model_equality/task_template.py` (full task statement).

## Infrastructure

`infra/`: shared job queue on one 4-GPU node of the JURECA supercomputer (`gpu_worker.py`, `GPU_PARTAGE.md`), with
the scripts that were run and their logs per GPU.

## Team

Clément, Orehn and Bruno, with coding agents (Devin, Claude, Codex) used to launch and monitor experiments.

## Language note

This README is in English. The working files written during the hackathon (coordination board, night plans, status
notes, some code comments) are kept as they were written, mostly in French.
