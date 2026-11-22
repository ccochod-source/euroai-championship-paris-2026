#!/usr/bin/env python3
"""Worker GPU PARTAGÉ de l'équipe : UN seul nœud (4 GPU) pour toute l'équipe.

Chaque GPU k lit son dossier  QUEUE/gpu<k>/  et exécute, un par un (ordre alphabétique), les scripts *.sh déposés,
avec CUDA_VISIBLE_DEVICES=<k>. Personne ne fait sbatch : on dépose un script, c'est tout.

  QUEUE/gpu<k>/*.sh            scripts en attente (déposer ici)
  QUEUE/gpu<k>/running/        script en cours
  QUEUE/gpu<k>/done|failed/    scripts terminés (code de sortie 0 / non nul)
  QUEUE/gpu<k>/logs/<nom>.log  sortie standard + erreurs du script
  QUEUE/STATUS.txt             état mis à jour toutes les 15 s
  QUEUE/STOP                   si présent : plus aucun nouveau lancement
  QUEUE/gpu<k>/KILL_<nom>      si présent : tue le script <nom> en cours sur ce GPU

Un script peut contenir une ligne « # TIMEOUT=<secondes> » (défaut 4 h).
Avant de lancer sur un GPU, le worker attend qu'aucun autre processus n'utilise ce GPU (transition propre).

    python3 gpu_worker.py --gpus 0,1,2,3 [--queue DIR]
"""
import argparse
import datetime
import os
import re
import signal
import subprocess
import time

DEFAULT_QUEUE = "/p/scratch/training2666/cochod2/euroai-paris/gpu_queue"
OWNERS = {0: "Model Equality", 1: "Model Equality", 2: "Watermark", 3: "Unlearning"}


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")


def busy_gpus():
    """Indices des GPU sur lesquels tourne déjà un processus de calcul (nvidia-smi)."""
    try:
        uu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=30).stdout
        idx = {u.strip(): int(i) for i, u in (l.split(",") for l in uu.strip().splitlines() if "," in l)}
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout
        return {idx.get(l.split(",")[0].strip()) for l in apps.strip().splitlines() if "," in l}
    except Exception:
        return set()


def say(msg):
    try:
        print(msg, flush=True)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--queue", default=DEFAULT_QUEUE)
    ap.add_argument("--until", default=None, help="heure de fin (epoch s) : plus de lancement 5 min avant")
    a = ap.parse_args()
    os.umask(0o002)
    gpus = [int(g) for g in a.gpus.split(",")]
    Q = a.queue
    for g in gpus:
        for sub in ("", "running", "done", "failed", "logs"):
            os.makedirs(os.path.join(Q, f"gpu{g}", sub), exist_ok=True)
    cur = {g: None for g in gpus}   # g -> (proc, name, t0, timeout, logfile)
    host = os.uname().nodename
    print(f"{now()} gpu_worker démarré sur {host}, GPU {gpus}, file {Q}", flush=True)
    last_status = 0
    while True:
        stop = os.path.exists(os.path.join(Q, "STOP")) or (a.until and time.time() > float(a.until) - 300)
        busy = None
        for g in gpus:
            d = os.path.join(Q, f"gpu{g}")
            c = cur[g]
            if c is not None:
                proc, name, t0, tmo, lf = c
                killf = os.path.join(d, f"KILL_{name}")
                rc = proc.poll()
                if rc is None and (time.time() - t0 > tmo or os.path.exists(killf)):
                    os.killpg(proc.pid, signal.SIGTERM)
                    time.sleep(5)
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                    rc = proc.wait()
                    try:
                        lf.write(f"\n[gpu_worker] tué ({'KILL' if os.path.exists(killf) else 'timeout'})\n")
                        if os.path.exists(killf):
                            os.remove(killf)
                    except OSError:
                        pass
                if rc is not None:
                    try:
                        lf.write(f"\n[gpu_worker] fin {now()} exit={rc} durée={time.time() - t0:.0f}s\n")
                        lf.close()
                        dst = "done" if rc == 0 else "failed"
                        os.replace(os.path.join(d, "running", name), os.path.join(d, dst, name))
                    except OSError as e:
                        say(f"{now()} gpu{g} fin {name} : écriture impossible ({e})")
                    say(f"{now()} gpu{g} fin {name} exit={rc}")
                    cur[g] = None
            if cur[g] is None and not stop:
                scripts = sorted(f for f in os.listdir(d) if f.endswith(".sh"))
                if not scripts:
                    continue
                if busy is None:
                    busy = busy_gpus()
                if g in busy:
                    continue  # un processus extérieur occupe encore ce GPU
                name = scripts[0]
                src = os.path.join(d, name)
                run = os.path.join(d, "running", name)
                try:
                    os.replace(src, run)
                except OSError:
                    continue
                txt = open(run, errors="replace").read()
                m = re.search(r"#\s*TIMEOUT=(\d+)", txt)
                tmo = int(m.group(1)) if m else 4 * 3600
                try:                              # quota disque plein : on lance quand même, sans journal
                    lf = open(os.path.join(d, "logs", name[:-3] + ".log"), "a", buffering=1)
                    lf.write(f"[gpu_worker] début {now()} sur {host} GPU {g}\n")
                except OSError as e:
                    say(f"{now()} gpu{g} journal impossible ({e}) : sortie vers /dev/null")
                    lf = open(os.devnull, "w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                proc = subprocess.Popen(["bash", run], cwd=d, stdout=lf, stderr=subprocess.STDOUT, env=env,
                                        start_new_session=True)
                cur[g] = (proc, name, time.time(), tmo, lf)
                say(f"{now()} gpu{g} début {name}")
        if time.time() - last_status > 15:
            last_status = time.time()
            lines = [f"gpu_worker sur {host} — mis à jour {now()}" + ("  [STOP : plus de nouveaux lancements]" if stop else "")]
            for g in gpus:
                d = os.path.join(Q, f"gpu{g}")
                n = len([f for f in os.listdir(d) if f.endswith(".sh")])
                c = cur[g]
                run = f"{c[1]} depuis {time.time() - c[2]:.0f}s" if c else "libre"
                lines.append(f"GPU {g} ({OWNERS.get(g, '?')}) : {run} ; {n} en attente")
            try:                                  # un quota disque plein ne doit jamais tuer le worker
                tmp = os.path.join(Q, ".STATUS.tmp")
                open(tmp, "w").write("\n".join(lines) + "\n")
                os.replace(tmp, os.path.join(Q, "STATUS.txt"))
            except OSError as e:
                say(f"{now()} STATUS non écrit : {e}")
        if stop and all(c is None for c in cur.values()):
            print(f"{now()} arrêt (STOP ou fin du job)", flush=True)
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
