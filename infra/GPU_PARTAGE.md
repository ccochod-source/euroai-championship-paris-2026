# GPU partagé de l'équipe — UN SEUL nœud pour TOUTE l'équipe

**Règle des organisateurs :** un seul nœud GPU (4 A100) à la fois pour toute l'équipe (ansellem1, cochod2, dasilvalopes3). Sinon, les jobs sont annulés et l'accès GPU est révoqué.

👉 **Ne faites JAMAIS `sbatch`, `srun` ni `salloc` vous-mêmes.** Annulez vos jobs existants : `scancel -u $USER`. Vérification : `squeue -u ansellem1,cochod2,dasilvalopes3` doit montrer **un seul job en cours (R)**.

## Le nœud
- Job actuel : `sim_worker` (15902386, ansellem1), jusqu'à **22h13**.
- Relais automatique : `team_gpu` (15902575, ansellem1), en attente avec une dépendance. Il démarre seulement après la fin du précédent (jamais 2 nœuds à la fois) et tourne jusqu'à ~12h demain. Le scheduler peut ajouter quelques minutes d'attente entre les deux : les scripts déposés restent en file et partent dès que le nœud est là.
- Sur le nœud, `gpu_worker.py` surveille un dossier par GPU et exécute les scripts déposés, un à la fois par GPU, avec `CUDA_VISIBLE_DEVICES` fixé.

| GPU | dossier | qui |
|---|---|---|
| 0 | `gpu_queue/gpu0/` | Model Equality |
| 1 | `gpu_queue/gpu1/` | Model Equality |
| 2 | `gpu_queue/gpu2/` | **Watermark** |
| 3 | `gpu_queue/gpu3/` | **Unlearning** |

`gpu_queue` = `/p/scratch/training2666/cochod2/euroai-paris/gpu_queue`

## Déposer un job (depuis le nœud de login)
1. Écrire un script bash. Il doit faire lui-même `cd` dans votre dossier et activer votre venv. Le GPU est déjà fixé : ne touchez pas à `CUDA_VISIBLE_DEVICES`, votre code voit un seul GPU (`cuda:0`).
2. Les nœuds de calcul n'ont **pas internet** : téléchargez modèles et données sur le nœud de login avant (`HF_HOME` partagé : `/p/scratch/training2666/cochod2/euroai-paris/.cache`), puis `HF_HUB_OFFLINE=1`.
3. Durée maximale facultative : une ligne `# TIMEOUT=<secondes>` (défaut 4 h).
4. Dépôt **atomique** : écrire sous un nom qui ne finit pas par `.sh`, puis `mv` (le worker ne prend que les `*.sh`). Les scripts partent par ordre alphabétique : préfixez avec un numéro.

Exemple (Watermark → GPU 2) :
```bash
Q=/p/scratch/training2666/cochod2/euroai-paris/gpu_queue
cat > $Q/gpu2/.tmp_wm <<'EOF'
#!/bin/bash
# TIMEOUT=3600
cd /p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1
python mon_script.py --args ...
EOF
mv $Q/gpu2/.tmp_wm $Q/gpu2/10_wm_run.sh
```

## Suivre
- État global : `cat $Q/STATUS.txt` (mis à jour toutes les 15 s : script en cours et taille de la file, par GPU).
- Logs : `tail -f $Q/gpu2/logs/10_wm_run.log` (sortie standard + erreurs, puis `exit=<code>` à la fin).
- Scripts terminés : `$Q/gpuK/done/` (code 0) ou `$Q/gpuK/failed/` (code non nul).
- Tuer votre script en cours : `touch $Q/gpu2/KILL_10_wm_run.sh` (nom exact du script).
- Annuler un script pas encore lancé : `rm $Q/gpu2/10_wm_run.sh`.
- Journal du worker : `$Q/worker.log`.

## À savoir
- Les scripts tournent sous l'utilisateur **ansellem1** (groupe training2666). Les fichiers créés sont accessibles en écriture au groupe (umask 002). Ne déposez que vos propres scripts.
- Un GPU libre n'est jamais pris par un autre : GPU 2 = Watermark, GPU 3 = Unlearning, même s'ils sont vides. Pour un échange temporaire, demandez à ansellem1 (ou déposez dans le dossier d'un autre GPU après accord).
- Python système = 3.9 ; utilisez votre propre venv.
