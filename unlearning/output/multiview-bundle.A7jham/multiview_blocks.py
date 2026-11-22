"""Job entry point: run the out-of-fold fine-tuning configurations for one GPU."""
import argparse, os, subprocess, sys
p = argparse.ArgumentParser()
p.add_argument("--depth", type=int)
args, _ = p.parse_known_args()
configs = {0: ["6:3e-5:3", "6:3e-5:5"], 2: ["12:3e-5:3", "6:3e-5:8"]}[args.depth]
job = os.environ.get("SLURM_JOB_ID", "local")
sys.exit(subprocess.run(["bash", "oof/run-sweep.sh", "output/oof-" + job, *configs]).returncode)

