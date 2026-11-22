# s3/seg — segment proposals as a prior for the watermark mask (running log)

Track goal: candidate regions for every image + measure how well they match the ground-truth masks (oracle), +
`snap(evidence128, proposals)` utility. GPU index 1. All numbers: 510 REAL training images, official `miou`.

## Files
- `seg/proposals.py` — computes the proposals (`./sg 1 seg/proposals.py <train|test> 16`).
- `seg/seglib.py` — `load_props`, `partitions`, `inst_cells`, `refine`, `grid`, oracle helpers, `snap`.
- `seg/oracle.py` — oracle test (`./sg 1 seg/oracle.py 16`), raw per-image results in `out/seg/oracle.json`.
- `seg/snaptest.py` — snap test with a degraded ground truth as evidence, `out/seg/snaptest.json`.

## Proposal format
`/p/scratch/training2666/dasilvalopes3/wm/s3/out/seg/<train|test>/<image_name>.npz`, one compressed npz per image
(~60 KB). Everything is at 256 x 256 (image downscaled 2x; 1 evaluation cell = 2 x 2 label pixels):

| key | shape / dtype | content |
|---|---|---|
| `slic` | (3,256,256) uint16 | SLIC superpixels, ~70 / ~265 / ~1030 segments (index 0 = coarsest) |
| `fh` | (3,256,256) uint16 | Felzenszwalb-Huttenlocher graph merging (on the MST of the Lab 8-neighbour graph), ~14 / ~66 / ~250 segments |
| `ws` | (2,256,256) uint16 | marker watershed, ~60 / ~275 segments |
| `ms` | (2,256,256) uint16 | mean-shift filtering + connected components, ~80 / ~340 segments |
| `inst_bits` | (N,8192) uint8 | N Mask R-CNN instance masks (score > 0.02, max 100, IoU>0.9 duplicates removed), bit-packed, sorted by decreasing score |
| `inst_score`, `inst_label`, `inst_box` | (N,), (N,), (N,4) | score, COCO category id, xyxy box in 512-px coordinates |

Label maps are partitions with labels 0..n-1, every segment 4-connected, small fragments merged.

Loading example (on the cluster, cwd = `wm/s3`, which is on PYTHONPATH):
```python
from seg.seglib import load_props, partitions, refine, snap
d = load_props("330fb70f8303", split="train")      # d["fh"][1], d["slic"][2], d["inst"] (N,256,256) bool ...
P = partitions(d)                                   # {"slic0".."slic2","fh0".."fh2","ws0","ws1","ms0","ms1","inst"}
best = refine(P["inst"], P["fh2"])                  # Mask R-CNN cells split by the fine FH partition
mask128 = snap(evidence128, [P["inst"], P["fh1"], P["fh2"]], margin=0.25)   # coarse -> fine, bool 128 x 128
```

## Log

### 1. Proposals computed for all 510 train + 1020 test images (check.py: 0 bad file)
- Runtime per image: Mask R-CNN 0.04-0.10 s on the A100 when it is free (0.45 s when GPU 1 was ~90% busy with other
  work), SLIC (GPU) 0.01 s, CPU part 0.59 s on one core (fh 0.22, mean-shift 0.21, SLIC clean-up 0.07, watershed 0.03,
  save 0.05). 2.5 GB of GPU memory. Whole set (1530 images) = about 5 minutes with 16 CPU workers. 105 MB on disk.
- Mask R-CNN weights: `maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth` downloaded on the login node from
  download.pytorch.org with `check_hash=True, weights_only=True` into `.cache/torch/hub/checkpoints`. Nothing else
  downloaded or installed. 27.7 instances per image on average (min 0 on one test image, max 100).
- Incidents: (a) job 15902267 was cancelled by another session in the middle of the train run, and `sg` was edited
  while a run was in flight; runs are resumable (existing files are skipped) and were completed. (b) a crash on the
  one image with zero detection, fixed.
- DEVIATION TO KNOW ABOUT: `sg` (srun -n 1) pins every step of every session to the same single core (cpu 48 on
  jrc0288; measured about 3x slower than an idle core: fh 0.7 s vs 0.22 s per image). For the CPU-bound classical methods my scripts call
  `os.sched_setaffinity` to move themselves to 16-32 idle cores at the top of the same node (240-255), still launched
  through `sg`, no extra srun/sbatch. Say so if this is not wanted; on the one shared core the 1530 images would take an estimated 50 min.

### 2. Oracle test (`oracle.py`, 510 real train images, official downsample_mask + miou)
Rule: a segment is kept when > 50% of its pixels are inside the ground truth; "+box" = segments first clipped to the
ground-truth bounding rectangle. Groups: full = 255 images (holes), shape = 136, rect = 119 (not-full).
`downsample_mask(gt512)` equals `masks_128.npz` on every cell. gridN = blind N x N squares (control: the oracle rises
mechanically with the number of segments, so a source is only informative if it beats the grid of similar #seg).

| source | #seg | full | shape | rect | shape+box | all |
|---|---|---|---|---|---|---|
| all-zeros / all-ones | 1 | .047 / .453 | .446 / .054 | .469 / .031 | | .252 / .248 |
| grid8 (control) | 64 | .706 | .728 | .658 | .839 | .701 |
| grid16 (control) | 256 | .830 | .837 | .776 | .889 | .819 |
| grid32 (control) | 1024 | .907 | .902 | .886 | .930 | .901 |
| grid64 (control) | 4096 | .953 | .950 | .943 | .961 | .950 |
| slic0 | 72 | .692 | .841 | .641 | .915 | .720 |
| slic1 | 270 | .808 | .911 | .750 | .949 | .822 |
| slic2 | 1033 | .898 | .942 | .874 | .965 | .904 |
| fh0 | 17 | .510 | .676 | .505 | .836 | .553 |
| fh1 | 75 | .588 | .828 | .572 | .911 | .649 |
| fh2 | 273 | .677 | .901 | .657 | .948 | .732 |
| ws0 | 61 | .665 | .797 | .639 | .891 | .694 |
| ws1 | 283 | .788 | .898 | .751 | .937 | .809 |
| ms0 | 79 | .598 | .842 | .585 | .921 | .660 |
| ms1 | 322 | .667 | .900 | .658 | .949 | .727 |
| inst (Mask R-CNN cells + background) | 134 | .525 | .746 | .528 | .892 | .585 |
| inst union, no background | 27 | .305 | .718 | .501 | .718 | .461 |
| inst+fh1 | 363 | .630 | .888 | .618 | .945 | .696 |
| inst+fh2 | 720 | .702 | .925 | .685 | .963 | .758 |
| inst+slic1 | 716 | .825 | .932 | .776 | .963 | .842 |
| inst+slic2 | 1698 | .905 | .951 | .883 | .972 | .912 |
| fh2+slic2 | 2208 | .909 | .953 | .887 | .973 | .915 |
| inst+fh2+slic2 | 2997 | .914 | .959 | .894 | .978 | .921 |
| inst+fh2+ms1+ws1+slic2 | 6097 | .930 | .968 | .913 | .983 | .936 |

(rect+box is 1.000 for every source by construction: it only says that the rectangle is known.)

Per watermark method (segments do not depend on the method, differences are small). full / shape / rect:

| source | pixelseal | trustmark | stegastamp |
|---|---|---|---|
| slic1 | .810 / .899 / .704 | .800 / .921 / .800 | .813 / .913 / .746 |
| fh2 | .677 / .895 / .635 | .653 / .898 / .672 | .701 / .912 / .664 |
| inst | .532 / .741 / .515 | .509 / .723 / .529 | .533 / .775 / .540 |
| inst+fh2 | .705 / .907 / .655 | .675 / .928 / .702 | .727 / .941 / .696 |
| inst+slic2 | .909 / .937 / .860 | .898 / .959 / .905 | .909 / .958 / .883 |
| grid16 (control) | .829 / .845 / .728 | .823 / .836 / .842 | .837 / .831 / .757 |

Reading:
- FULL images (holes): image segments are WORSE than blind squares of the same count (fh2 .677, ms1 .667, slic1 .808
  vs grid16 .830; Mask R-CNN cells .525, barely above all-ones .453; only 1% of full images above 0.9 with inst).
  The holes are NOT aligned with the content of the image. Visual check (`out/seg/viz_full.jpg`, 7 images): the
  holes are smooth blobs / silhouettes that cross the objects of the picture at random (a cone shape on a ski slope,
  blobs over a face...). The README sentence "holes follow the contours of real objects" does not hold for holes.
  No segmenter can help there (Segment Anything included).
- RECT images: same thing, as expected (an invisible rectangle is not an image edge): segments <= grid.
- SHAPE images (27% of train): masks ARE aligned with the picture (`out/seg/viz_shape.jpg`: a slice of cake, a
  scooter, a traffic light, a logo on a shirt). slic1 .911 / fh2 .901 / ms1 .900 against .837 for the grid of the
  same count; with the bounding rectangle known, inst+fh2 reaches .963 and 93% of the shape images are above 0.9
  (fh1 alone, 75 segments: .911, 71% above 0.9). Mask R-CNN alone is not enough (.746; .892 with the box): many
  shapes are not COCO objects or are parts of objects; it adds +.02 on top of a classical partition (with more segments, so part of that gain is mechanical).

### 3. `snap` (`seglib.snap`, `seglib.snap_fit`; test in `snaptest.py`)
Evidence = ground truth averaged on 16 x 16 cells + Gaussian noise (sigma per cell), bilinearly upsampled to 128.
Synthetic evidence: real evidence maps will have other error structures; only 2-3 settings were tried, on the same
510 images.

| variant | sigma 0: full / shape / rect | sigma 0.15 | sigma 0.3 |
|---|---|---|---|
| evidence > 0.5, no snap | .870 / .875 / .807 | .856 / .854 / .794 | .765 / .782 / .718 |
| hard snap slic1 | .778 / .867 / .713 | .772 / .864 / .713 | .754 / .831 / .704 |
| hard snap slic2 | .833 / .876 / .778 | .826 / .862 / .774 | .772 / .804 / .727 |
| hard snap fh2 | .656 / .854 / .632 | .651 / .847 / .627 | .648 / .834 / .623 |
| hard snap inst+fh2 | .674 / .867 / .647 | .672 / .858 / .641 | .663 / .840 / .637 |
| hard snap grid16 (control) | .806 / .821 / .747 | .802 / .814 / .748 | .768 / .783 / .721 |
| soft snap (margin .25) slic1 | .852 / .877 / .801 | .840 / .861 / .788 | .770 / .797 / .720 |
| soft snap (margin .25) inst+fh2 | .779 / .869 / .732 | .771 / .852 / .722 | .738 / .816 / .685 |
| multi-scale (margin .35) slic0>slic1>slic2 | .839 / .875 / .788 | .826 / .856 / .773 | .769 / .811 / .729 |
| snap_fit slic1 | .806 / .907 / .757 | .777 / .849 / .719 | .654 / .686 / .600 |
| snap_fit inst+slic1 | .816 / .916 / .769 | .780 / .852 / .725 | .654 / .681 / .597 |
| snap_fit grid32 (control) | .861 / .866 / .824 | .846 / .848 / .798 | .698 / .715 / .653 |

Reading:
- Snapping only pays on SHAPE images, and mostly when the evidence is noisy: +.058 at sigma 0.3 (hard, inst+fh2:
  .782 -> .840), +.010 at sigma 0.15 (hard, slic1), 0 at sigma 0 with the plain snap. The oracle (.93-.96) is far
  from being reached: a 16 x 16 evidence map cannot decide small segments near the boundary.
- `snap_fit` (bounded least squares: the segment mask, re-averaged on the 16 x 16 cells, must reproduce the
  evidence) gets +.04 on shapes with clean evidence (.875 -> .916) but is fragile: it loses to no-snap from
  sigma 0.15 on. Use it only if the evidence is clean and its true resolution is known. It is slow (~1 s/image).
- On FULL and RECT images every snap variant is at or below the un-snapped evidence (best case +.007 at sigma 0.3,
  which a blind grid also gets: plain denoising). Do not snap those.

### 4. Recommendation
- Use segments only for not-full images whose mask is not a plain rectangle. Source: `refine(P["inst"], P["fh2"])`
  for a hard snap on noisy evidence, `slic1` (or `inst+slic1`) otherwise; clip to the estimated rectangle when one
  is known (the oracle with the box is .95-.96).
- Holes in full images need another idea (they are not image regions): shape priors on smooth blobs, not segments.
- Segment Anything: not worth the approval process now. It cannot help holes or rectangles (73% of the data), and
  on shapes the already-installed proposals reach an oracle of .95-.96 with the box; the limiting factor there is
  the evidence, not the segments. Even with perfect evidence, perfect segments could add at most (1 - .963) x 27%
  = +.01 overall with the box known (+.02 without: 1 - .925).
