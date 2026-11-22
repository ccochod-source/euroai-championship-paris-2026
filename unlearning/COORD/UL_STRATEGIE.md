# UNLEARNING — PLAN TO GET PAST 0.12 (research summary, 22 Nov ~00:00, Claude/Orehn)
Diagnosis: a single global A for all images => impossible to reproduce (1) the STD of the gold model's loss
(gold = model that never saw the image: most images stay easy/correct class, a tail of hard images
has a high loss) and (2) WHICH wrong class the gold model predicts per image (prediction_score, KL T=2).

## Quick checks (10 min)
1. Histogram of pseudo-labels (argmax of the original model on forget). Over-represented classes (>>35/class)
   => the gold model partly forgets these classes: lower their mass separately.
2. Public init? cosine of patch_embed/block0 vs timm deit3_base_patch16_224 / vit_base_patch16_224 (augreg).
   If it matches => the gold model was fine-tuned from there => cheap "retrain" possible from the same init.

## Ranked strategies
1. OUT-OF-FOLD TEACHERS (priority 1). K=5 folds of the 3500 images. For fold k: train on the 4 other folds
   (pseudo-labels, label smoothing 0.1), predict fold k => "never seen" logits z_oof.
   - fast version: frozen DINOv2-B/14 or CLIP-L/14 + logistic regression (minutes).
   - strong version: public init (if found) or re-init of blocks 10-11 + head; AdamW lr 1e-5..3e-5 blocks, 1e-4 head, 5-15 epochs.
   - average 2-3 seeds/splits (the gold model is an average of models).
   - CALIBRATE: target z_t = beta*z_oof/T + (1-beta)*(z_orig with top-1 lowered). Choose T, beta so that
     the mean AND std of the CE (pseudo-labels) on forget = target "unseen" statistics.
2. PER-IMAGE A (fallback): difficulty d_i = mean rank of (leave-one-out DINOv2 kNN: fraction of neighbours
   from another class) + (confidence drop under strong augmentation/noise) + (OOF probability of the pseudo-label).
   A_i = A_max*sigmoid((d_i-tau)/w). Freed mass -> ranks 2-5 of the original in proportion to their softmax
   (NOT uniform). Effect: bimodal loss std, far less accuracy loss than a global A=0.05.
3. RETENTION TERM on substitute data: L = L_forget + lambda*KL(T=2)/MSE of centred logits vs original
   on NON-forget images (ImageNet val/COCO/generated images), lambda 0.5-2. Directly targets retain_score
   (KL(p_orig||p_sub) on control) => allows stronger forgetting without dropping below 70%.
   (Precedent NeurIPS 2023: Amnesiacs, 8th place "Forget" — MSE > CE.)
4. ALTERED COPIES = 50% of the forget score: train on K augmented views per image (RandomResizedCrop 0.5-1,
   ColorJitter, blur, JPEG 30-70, flip, small rotation); with #1 compute the target per view.
5. Match the LOGITS (MSE of centred logits) + small KL at T=1 and T=2. When in doubt about the wrong class:
   spread over the original's top-3/5 (KL(gold||sub) punishes p_sub~0).
6. Parameters: blocks 8-11 + LayerScale + head (optional SalUn mask, top 10-50% |grad|). AdamW lr 5e-6..3e-5,
   3-10 epochs, cosine, checkpoint per epoch; pick the checkpoint whose mean forget loss ~ unseen target (SCRUB+R).
   Alternative LoTUS (CVPR'25): automatic temperature exp(2*(Acc_forget-Acc_unseen)), ViT lr 1e-6, 3 epochs.
7. Reset/noise on the last 1-2 blocks + repair by distillation on substitutes (only if the substitutes are good).
8. TO AVOID: Bad Teacher, UNSIR/max-error noise, random relabelling, NegGrad+, SSD (they push towards uniform/random,
   the opposite of the gold model's structure).

## BISECTION WITHOUT OVERFITTING
- 1 single knob s (A_max, or beta/T). LOCAL proxies independent of the training teacher:
  mean_std proxy (mean/std CE on forget+views vs unseen statistics), KL T=2 vs ANOTHER teacher (DINOv2 if we trained on fine-tuned OOF, or the reverse),
  retain proxy (accuracy ratio + KL on substitutes). GUARD: estimated accuracy >= 75%.
- Bisection on s until mean loss = target, then golden-section on log s in [s*/2, 2s*].
- Noise: bootstrap of the proxies on random 30% subsets => a gap < noise is not a signal.
- Submit 3 models at most: s*, 0.7s*, 1.4s*. Final = best proxy among those within the noise of the best LB score.
- Keep the A-only pipeline as a fallback. Respect SUBMIT_LOCK + 10 min.

Sources: arxiv 2406.09073 (NeurIPS 2023 MU challenge), SCRUB 2302.09880, SalUn 2310.12508, LoTUS 2503.18314,
JiT 2402.01401, RUM 2406.01257, junhyunb.github.io/en/projects/neurips2023-unlearning/
