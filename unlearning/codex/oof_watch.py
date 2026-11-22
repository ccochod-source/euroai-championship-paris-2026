"""Finite coordinator: idle worker GPU2/3 only; CPU fit with two threads.

No Slurm commands, no leaderboard, no credentials. At most three predeclared
training strengths. Existing queued jobs and teammate files are untouched.
"""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;QUEUE=ROOT.parent/'gpu_queue'
TZ=ZoneInfo('Europe/Paris');STATE=HERE/'oof_state.json'
MARK='<!-- CODEX_OOF_BEGIN -->';END='<!-- CODEX_OOF_END -->'
CONFIGS=[('oof_calibrated_s1',1.),('oof_calibrated_s07',.7),('oof_calibrated_s14',1.4),('oof_adaptive_s1',1.)]
def now():return dt.datetime.now(TZ)


def atomic_json(p,obj):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));temp.replace(p)


def record(line):
    with (ROOT/'COORD/BOARD.md').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(f'\n- {now():%d/%m %H:%M} — Codex OOF : {line}\n')


def free_gpu():
    p=QUEUE/'STATUS.txt'
    if not p.exists() or time.time()-p.stat().st_mtime>30 or (QUEUE/'STOP').exists():return None
    txt=p.read_text()
    for g in (3,2):
        folder=QUEUE/f'gpu{g}'
        if re.search(rf'^GPU {g} .*: libre ; 0 en attente$',txt,re.M) and not list(folder.glob('*.sh')) and not list((folder/'running').glob('*.sh')):return g
    return None


def current_best():
    rows=[]
    for line in (ROOT/'COORD/SUBMISSIONS.md').read_text().splitlines():
        cols=[s.strip() for s in line.split('|')]
        if len(cols)<8 or not cols[4].lower().startswith('soumis'):continue
        m=re.search(r'\d+[,.]\d+',cols[5]);path=cols[3].strip('`')
        if m and path.startswith(('orehn/ckpt/','codex/ckpt/')) and (ROOT/path).is_file():rows.append((float(m.group().replace(',','.')),path))
    return max(rows)[1] if rows else 'orehn/ckpt/clem_A03r3_g4.pt'


def queue_job(state,name,cmd):
    g=free_gpu()
    if g is None:return 'Attente GPU2/3 libre ; CPU priorite1 prepare, aucun job Slurm ajoute'
    p=HERE/'jobs'/f'{name}.sh';p.parent.mkdir(exist_ok=True)
    p.write_text('#!/bin/bash\n# TIMEOUT=3600\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+
                 '\nexport OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1\n'+
                 shlex.join([str(ROOT/'.venv/bin/python'),'-u',*cmd])+'\n')
    qname='90_codex_'+name+'.sh'
    os.link(p,QUEUE/f'gpu{g}'/qname)
    state['jobs'].append({'name':name,'gpu':g,'queue':qname,'time':now().isoformat(),'status':'queued'})
    record(f'GPU{g} observe libre et file vide ; depot {qname}. Sorties exclusivement codex/ ; aucun envoi officiel.')
    return f'{name} depose GPU{g}'


def advance(state):
    for j in state['jobs']:
        for status,folder in [('running','running'),('failed','failed'),('done','done')]:
            if (QUEUE/f"gpu{j['gpu']}"/folder/j['queue']).exists():j['status']=status
        if j['status'] in ('queued','running'):return f"{j['name']} : {j['status']} GPU{j['gpu']}"
        if j['status']=='failed':return f"ECHEC {j['name']} ; inspection manuelle requise, pas de relance aveugle"
    if not (HERE/'cache/oof_features.pt').exists():
        if any(j['name']=='oof_extract' for j in state['jobs']):return 'Extraction terminee sans cache ; inspection requise'
        return queue_job(state,'oof_extract',[str(HERE/'oof_extract.py')])
    if not (HERE/'cache/oof_targets.pt').exists():
        if 'cpu_fit_pid' in state:
            try:os.kill(state['cpu_fit_pid'],0);return 'CPU2 : 15 professeurs OOF + calibration moyenne/std en cours'
            except ProcessLookupError:return 'ECHEC CPU OOF sans cible ; consulter logs/oof_fit.log'
        env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        log=(HERE/'logs/oof_fit.log').open('a')
        proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(HERE/'oof_fit.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['cpu_fit_pid']=proc.pid;log.close();record('Features DINO10vues pretes ; regression logistique 5folds x3partitions lancee CPU2. Prof evaluation reserve.')
        return 'CPU2 OOF demarre'
    if not (HERE/'cache/oof_adaptive_targets.pt').exists() and 'adaptive_pid' not in state:
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        log=(HERE/'logs/oof_adaptive.log').open('a')
        proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(HERE/'oof_adaptive.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['adaptive_pid']=proc.pid;log.close()
    for name,strength in CONFIGS:
        gate=HERE/'logs'/f'{name}_gate.json'
        if gate.exists():continue
        if any(j['name']==name for j in state['jobs']):return f'{name} fini sans rapport de selection ; inspection requise'
        target='oof_adaptive_targets.pt' if name=='oof_adaptive_s1' else 'oof_targets.pt'
        if not (HERE/'cache'/target).exists():return 'Attente cibles A par image CPU2 ; consulter log si echec'
        return queue_job(state,name,[str(HERE/'oof_train.py'),'--name',name,'--strength',str(strength),'--targets',target,'--baseline',current_best()])
    return 'Trois forces OOF + priorite2 A/image terminees ; selection locale disponible, aucune soumission par ce programme'


def refresh(state,phase):
    reports=[]
    for name,s in CONFIGS:
        p=HERE/'logs'/f'{name}_gate.json'
        if p.exists():
            d=json.loads(p.read_text());reports.append(f"- `{name}` : retain_est={d['retain_est']:.4f}, delta proxy IC95={d['proxy_delta_95_interval']}, candidat local admissible={d['local_candidate_eligible']}.")
    cpu=HERE/'logs/oof_cpu_clean.log'
    cpu_tail=cpu.read_text().splitlines()[-1] if cpu.exists() and cpu.stat().st_size else 'pas de journal CPU'
    text=f'''{MARK}
# Resume de nuit pour 7h40 — priorite OOF

Mise a jour : {now():%d/%m %H:%M:%S %Z}. **{phase}**.
Repli CPU2 vue propre (borne20min) +15profs : `{cpu_tail[:300]}`.

- Plan UL_STRATEGIE : DINOv2-B14 gele, 5 plis x 3 partitions, pseudo-labels original, label smoothing0.1 ; cibles par vue et calibration moyenne/std CE.
- Statistiques unseen estimees par les probes retain-only6/7 ; evaluation par probe8 distinct, jamais utilise pour calibrer. Ce sont des approximations, pas les gold inconnus.
- FT blocs9-11 + head/norm, 1100pas, lr3e-5, KL retain poids2 + MSE logit centre ; 7 vues train. Shift/rotation/bruit reserves au test. Trois forces fixees d'avance : 1, 0.7, 1.4.
- Comparaison au meilleur checkpoint officiel disponible au lancement de chaque essai : `{current_best()}` actuellement. Refaire la comparaison sil change avant un envoi.
- Garde locale : retain_est>=0.75, amelioration proxy independant sur sous-ensembles30% (IC95 positif), KL des vues reservees amelioree. retain_est est accord/(1+KL) sur2000substituts sans labels, pas une accuracy mesuree.
- Soumissions Codex : aucune nouvelle depuis ft2_A03=0.0735. Ce coordinateur ne soumet jamais ; tout envoi exige controles actuels SUBMIT_LOCK, cooldown equipe10min et max3.
- CPU2 ; GPU2/3 uniquement si libres et files vides, via worker existant. Aucun sbatch/scancel, tmux, runner ME ou fichier de coequipier touche. Priorite2 A/image preparee et testee apres les trois forces OOF (4 essais locaux, plafond3 soumissions).

{chr(10).join(reports) if reports else 'Resultats a venir dans codex/logs/*_gate.json et COORD/RESULTS.csv.'}

Arret borne a07h35. Le precedent watcher repair est arrete via NIGHT_STOP. Artefacts : oof_features.pt, oof_targets.pt, oof_calibration.json, *_oof_eval.json, *_gate.json ; seuls les checkpoints finis sont utilisables.
{END}
'''
    p=HERE/'STATUS.md';old=p.read_text() if p.exists() else ''
    if MARK in old and END in old:old=old.split(END,1)[1].lstrip()
    tmp=HERE/'.STATUS.oof.tmp';tmp.write_text(text+'\n'+old);tmp.replace(p)
    state.update(updated=now().isoformat(),phase=phase);atomic_json(STATE,state)


def main():
    lease=(HERE/'oof_watch.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads(STATE.read_text()) if STATE.exists() else {'jobs':[]}
    (HERE/'oof_watch.pid').write_text(str(os.getpid()))
    while now()<dt.datetime(2026,11,22,7,35,tzinfo=TZ):
        if (HERE/'OOF_STOP').exists():refresh(state,'Arret demande via OOF_STOP');return
        try:refresh(state,advance(state))
        except Exception as e:
            print(now().isoformat(),repr(e),flush=True)
            try:refresh(state,'Erreur coordinateur : '+repr(e))
            except OSError:pass  # Shared quota can temporarily prevent status writes.
        # Also reap completed CPU children to avoid treating a zombie as live.
        try:os.waitpid(-1,os.WNOHANG)
        except ChildProcessError:pass
        time.sleep(15)
    refresh(state,'Fin de surveillance07h35 ; consulter les resultats ci-dessus')


if __name__=='__main__':main()
