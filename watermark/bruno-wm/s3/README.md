# s3 — training-free "propose and verify" track (session epitech-hack-meta-ee)

Hackathon task (deadline 22 Nov 12:00): for each 512x512 test image, output a 128x128 binary mask (1 cell = 4x4 px) of
the pixels that still carry an invisible watermark. We know, for every image, the method (pixelseal / trustmark /
stegastamp) and the embedded message. Score = mean over images of (IoU of class 0 + IoU of class 1) / 2; an empty
union counts as IoU 1. Predicting all-0 or all-1 gives about 0.45 on average. Best team: 0.63. Us: 0.44.

## Facts measured on the 510 training images (170 per method)
- `full_watermark=true` (half of the images, flag only given on train): the watermark covers the whole image except
  one or more HOLES. Mask bounding box = whole image in 98-99% of them. Hole area: median 6% of the image
  (5%/25%/75%/95% quantiles: 0.5% / 2.3% / 12% / 29%). All-ones scores 0.455 on these.
- `full_watermark=false` (other half): watermark covers a small part (median 4-5% of the image). 47% are exact
  axis-aligned RECTANGLES (short side: median 60 px, 5% quantile 16 px), the rest are SHAPES: an object silhouette,
  or a rectangle cut by an object. All-zeros scores about 0.46 on these.
- Holes and shapes follow the contours of real objects of the picture (a slice of cake, a sign, a person, a patch
  of a vase...) or are blob-like regions. Images are AI-generated photos (COCO-like objects).
- The watermark was embedded on a rectangle (the "frame"): the frame was resized to the model's working resolution
  (256 pixelseal, 245/256 trustmark, 400 stegastamp), watermarked, resized back. For full images the frame is the
  whole image. The official decoder reads the message (bit accuracy ~1.0, margin 6-10) only when it is shown
  (almost) exactly the frame. On real full images: margin = mean over bits of logit*sign is ~8.7 with 4% holes,
  ~4.3 with 23% holes, ~2.2 with 35% holes (trustmark).
- IMPORTANT sim-to-real gap: for trustmark and pixelseal, the watermark in the challenge images is NOT pixel-wise
  correlated with what our embedder (`wm.embed`) produces for the same image and message, although the decoder
  reads both equally well. Networks trained on our own embeddings fail on the real images. So: validate everything
  on the REAL training images, never on images we watermark ourselves.
- Another session got 0.81 (out-of-fold) on stegastamp by fine-tuning a U-Net on the real training images; another
  one finds plain rectangles for trustmark by searching the crop that maximises decoding (0.94 on plain rectangles).
  Do not redo those. Open problems: HOLES in full images and SHAPES, above all for trustmark and pixelseal.

## How to run code
- Edit files locally in `/Users/brunodasilvalopes/Hack_EPITECH_META/EPITECH_HACK_META/s3/<your subdir>/`, copy with
  `scp -q <files> juelich:/p/scratch/training2666/dasilvalopes3/wm/s3/<your subdir>/`, run with
  `ssh -o BatchMode=yes juelich 'cd /p/scratch/training2666/dasilvalopes3/wm/s3 && ./sg <gpu> <subdir>/<script.py> [args]'`.
  `sg` runs the script on GPU <gpu> (0-3) of one of our running Slurm jobs. It prints "NO RUNNING JOB" if none.
- `s3/common.py` (already on the cluster, on PYTHONPATH): `load_wm(method)`, `load_items(split, method)`,
  `gt128()`, `gt512(name)`, `load_img(path)`, `sm_of(bits)`, `signed(wm, imgs, sm)` -> (margin, bit accuracy) for a
  batch of images of any size, `miou(pred128, gt128)` (official metric), `downsample_mask(mask512)` (official),
  `kind(gt)`. Kit wrappers: `/p/scratch/training2666/dasilvalopes3/wm/kit/wmloc/watermarks.py` (`wm.logits`,
  `wm.features`, `wm.embed`; trustmark decoder = `wm.net`, a torch ResNet-50 on 245x245 inputs scaled to [-1,1];
  pixelseal decoder = `wm.det` on 256x256 inputs in [0,1], float32 only; stegastamp = ONNX, not differentiable).
- Write outputs under `/p/scratch/training2666/dasilvalopes3/wm/s3/out/`.

## Rules (hard)
- The GPUs are SHARED with other running work of the same user: at most ~12 GB of GPU memory, no multi-hour runs,
  use only the GPU index you were given.
- Never run sbatch / scancel / kill. Never modify or delete anything outside `.../wm/s3/` on the cluster or outside
  `s3/` locally (other sessions own `wm/`, `solution/`, `bruno/`). Reading is fine.
- No leaderboard submission. Never print or copy the API key.
- No pip install, git clone or download of third-party code or models without reporting back first (the user's
  security rule requires vetting). Already installed: torch 2.5.1, torchvision 0.20.1, timm, opencv (cv2), scipy,
  onnxruntime, safetensors. NOT installed: transformers, scikit-image, scikit-learn, segment_anything, diffusers.
- Compute nodes have no internet.
- Measure on the real training images with the official metric; report honest numbers, including failures.
