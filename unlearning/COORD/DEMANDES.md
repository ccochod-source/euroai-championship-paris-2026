# REQUESTS — runs that Clément (or his session) asks Orehn's session to launch

File created on 22 Nov at 01:30 by Orehn's session (it did not exist): Clément, correct the rules below if they do not suit you.

## Rules
1. One request = one row of the table, status **NEW**. Only Orehn's session changes the status: NEW → TAKEN → RUNNING → DONE (or REFUSED, with the reason).
2. The command must be a call to an existing script in the `unlearning/` folder (e.g. `orehn/src/finetune.py --name ... --A ...`), write only in `unlearning/orehn/ckpt`, `unlearning/output` or your own folder, and last less than 40 min. No sbatch, srun, scancel, no deleting other people's files, no token.
3. Launch through `gpu_queue/gpuN/` with the existing worker (a single GPU job for the team). GPU 3 by default; GPU 0 only if it is free on the Model Equality side.
4. Orehn's session reads this file every 5 min, gives priority to your requests when the GPU is contended (alternating with its own runs), then fills in log + results here and in RESULTS.csv.
5. No leaderboard submission from this file: that needs a "requested" line in SUBMISSIONS.md and the SUBMIT_LOCK.

## Requests
| No. | Time | Requester | Command (finetune.py arguments or script) | Desired GPU | Status | Log | Results (mean CE / std, acc, retain_est) |
|---|---|---|---|---|---|---|---|
