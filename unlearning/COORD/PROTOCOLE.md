# PROTOCOL — unlearning coordination (Orehn, Clément = cochod2, Bruno = dasilvalopes3)

1. **Read `BOARD.md` before starting a track.** Never take a track that is already taken (status "in progress" or "done"). For a close variant: talk to its owner first.
2. **Register in `BOARD.md` before launching**: who, track, status, folder, GPU/node.
3. **Always add your results** to `RESULTS.csv` (one line per run, even failed ones), with the same local metrics (below).
4. **Submission**: never without human approval. Write the `SUBMIT_LOCK` file (who, time, checkpoint), announce it in `SUBMISSIONS.md`, wait for approval, submit, record the score, delete `SUBMIT_LOCK`. If `SUBMIT_LOCK` already exists: wait. 10 min between submissions.
5. **GPU**: a single node for the whole team, no personal `sbatch` without human approval. Everyone writes only in their own folder (`orehn/`, `oof/`, `eugenia-trials/`, `output/bruno/`...) and in `COORD/`.
6. Checkpoints: float32, same architecture, `strict=True`, < 450 MB. Disk: delete useless intermediate checkpoints (the GPU worker crashed at 20:46 on "Disk quota exceeded").

## Shared local metrics (columns of RESULTS.csv)
Label y = argmax of the original model on the clean image (it memorised everything: confidence 0.9997).
- `f_acc`, `f_Lmean`, `f_Lstd`: agreement with y, mean and standard deviation of the cross-entropy (T=1) on the clean forget images. Original: 1.000 / 0.000 / 0.002.
- `alt_acc`, `alt_Lmean`, `alt_Lstd`: same on an altered copy (x·0.8+0.1 in [-1,1] space, "bright" view). Original: 0.846 / 0.61 / 1.71.
- `ctl_KL`, `ctl_agree`: KL(original || candidate) at T=2 and argmax agreement on control images (say which ones in `notes`).
- `rel_dw`: ||θ − θ0|| / ||θ0||.
- `score_officiel`: leaderboard score if submitted, otherwise empty.
