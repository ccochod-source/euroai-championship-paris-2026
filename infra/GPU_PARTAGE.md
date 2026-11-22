# Shared team GPU — ONE SINGLE node for the WHOLE team

**Organisers' rule:** a single GPU node (4 A100) at a time for the whole team (ansellem1, cochod2, dasilvalopes3). Otherwise jobs are cancelled and GPU access is revoked.

👉 **NEVER run `sbatch`, `srun` or `salloc` yourself.** Cancel your existing jobs: `scancel -u $USER`. Check: `squeue -u ansellem1,cochod2,dasilvalopes3` must show **a single running job (R)**.

## The node
- Current job: `sim_worker` (15902386, ansellem1), until **22:13**.
- Automatic relay: `team_gpu` (15902575, ansellem1), pending with a dependency. It starts only after the previous one ends (never 2 nodes at once) and runs until ~12:00 tomorrow. The scheduler may add a few minutes of waiting between the two: queued scripts stay in the queue and start as soon as the node is there.
- On the node, `gpu_worker.py` watches one folder per GPU and runs the scripts dropped there, one at a time per GPU, with `CUDA_VISIBLE_DEVICES` set.

| GPU | folder | who |
|---|---|---|
| 0 | `gpu_queue/gpu0/` | Model Equality |
| 1 | `gpu_queue/gpu1/` | Model Equality |
| 2 | `gpu_queue/gpu2/` | **Watermark** |
| 3 | `gpu_queue/gpu3/` | **Unlearning** |

`gpu_queue` = `/p/scratch/training2666/cochod2/euroai-paris/gpu_queue`

## Dropping a job (from the login node)
1. Write a bash script. It must itself `cd` into your folder and activate your venv. The GPU is already set: do not touch `CUDA_VISIBLE_DEVICES`, your code sees a single GPU (`cuda:0`).
2. Compute nodes have **no internet**: download models and data on the login node beforehand (shared `HF_HOME`: `/p/scratch/training2666/cochod2/euroai-paris/.cache`), then `HF_HUB_OFFLINE=1`.
3. Optional maximum duration: a line `# TIMEOUT=<seconds>` (default 4 h).
4. **Atomic** drop: write under a name that does not end in `.sh`, then `mv` (the worker only picks up `*.sh`). Scripts start in alphabetical order: prefix with a number.

Example (Watermark → GPU 2):
```bash
Q=/p/scratch/training2666/cochod2/euroai-paris/gpu_queue
cat > $Q/gpu2/.tmp_wm <<'EOF'
#!/bin/bash
# TIMEOUT=3600
cd /p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1
python my_script.py --args ...
EOF
mv $Q/gpu2/.tmp_wm $Q/gpu2/10_wm_run.sh
```

## Monitoring
- Global state: `cat $Q/STATUS.txt` (updated every 15 s: running script and queue size, per GPU).
- Logs: `tail -f $Q/gpu2/logs/10_wm_run.log` (stdout + stderr, then `exit=<code>` at the end).
- Finished scripts: `$Q/gpuK/done/` (code 0) or `$Q/gpuK/failed/` (non-zero code).
- Kill your running script: `touch $Q/gpu2/KILL_10_wm_run.sh` (exact script name).
- Cancel a script not yet started: `rm $Q/gpu2/10_wm_run.sh`.
- Worker log: `$Q/worker.log`.

## Good to know
- Scripts run as user **ansellem1** (group training2666). Created files are group-writable (umask 002). Only drop your own scripts.
- A free GPU is never taken by someone else: GPU 2 = Watermark, GPU 3 = Unlearning, even when empty. For a temporary swap, ask ansellem1 (or drop into another GPU's folder after agreement).
- System Python = 3.9; use your own venv.
