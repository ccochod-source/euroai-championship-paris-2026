# Watermark — Orehn/simulator session (22 Nov, 10:46–10:56)

## Result: out/sub_orehn_a3.pt — out-of-sample mIoU (train, real masks) 0.7619
File: /p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task/orehn/out/sub_orehn_a3.pt (1020 masks 128x128 bool, format checked). Report: out/sub_orehn_a3.json.
**Not submitted: I have no submission key (no ~/.hack_api_key for ansellem1). To be submitted by Bruno or Orehn:**
  /p/scratch/training2666/dasilvalopes3/wm/submit.sh /p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task/orehn/out/sub_orehn_a3.pt

## What was done (read-only on Bruno's files)
1. Re-assembly with the networks finished last night (out-of-sample probabilities from wm/out): pixelseal ftP6P (0.6419, threshold 0.35),
   trustmark ftT6P (0.7344, threshold 0.75), stegastamp mean of ftP + ftS5P (0.8830, threshold 0.45). Overall 0.7531. (assemble2.py)
2. Morphological post-processing, per-regime thresholds, components, bounding boxes: no gain in cross-validation, not applied. (assemble3.py)
3. Fusion with Bruno's rectangle search (network OR rectangle when acc >= 0.8, non-full images): (assemble4.py)
   trustmark 0.7344 -> 0.7546 (2-half cross-validation; 83 images with a rectangle: 0.764 -> 0.806): applied
   pixelseal 0.6419 -> 0.6472 (20 images with a rectangle: 0.819 -> 0.873): applied
   stegastamp 0.8830 -> 0.8858: below the 0.004 threshold, not applied
   Expected overall 0.7619 (+0.009).

## Limiting factor
PixelSeal (0.64; full images with holes 0.60, partial 0.68), then TrustMark full images (0.70). StegaStamp is at 0.88.

## Caveat
I do not know which submission gives the current public 0.75 nor its local score: if it already includes these networks AND the rectangle fusion, the gain will be zero.

## Public result (11:00): sub_orehn_a3.pt -> 0.7448, BELOW Bruno's best (0.7589, kept)
- The local estimate (0.7619 out-of-sample on train) did not transfer. Difference with Bruno's file from yesterday 20:50 (sub_s3_v8.pt): only 0.816 agreement between the two sets of masks, so two genuinely different solutions.
- Search for an addition starting from Bruno's file (11:00–11:05): the script that built sub_s3_v8.pt is not on the cluster (no reference to sub_s3 in wm/s3; assembled locally by Bruno) and there are no matching TRAIN masks. So I cannot validate locally any modification of his file (fusion with my networks, rectangles, vote). Without validation, no 2nd submission: we stop here, the retained submission stays Bruno's.
- Lessons: (1) my out-of-sample score came from networks whose thresholds and ensembles were chosen on the same 510 images, with no control set; (2) the public split (~300 images) has a noise of about 0.01, the measured gap (−0.014) is small but goes the wrong way; (3) to compare two solutions we would have needed the train masks of Bruno's pipeline.
