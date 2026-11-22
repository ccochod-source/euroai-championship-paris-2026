"""Finite overnight coordinator, stdlib-only CPU process on the login node.

Queues bounded, predeclared repair experiments on GPU3 only when free. It never
submits: current validation controls have no true labels, so the >=75% accuracy
condition cannot be certified. No scheduler, credentials, or process control.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import time
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
QUEUE=ROOT.parent/'gpu_queue'
TZ=ZoneInfo('Europe/Paris')
START='<!-- CODEX_NIGHT_BEGIN -->'
END='<!-- CODEX_NIGHT_END -->'
STATE=HERE/'night_state.json'
CONFIGS=[{'tag':'last4','last':4,'anchor':8,'steps':900,'lr':2e-5},
         {'tag':'full','last':12,'anchor':16,'steps':1200,'lr':1e-5}]


def now():return dt.datetime.now(TZ)


def save_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    os.replace(temp,path)


def scores():
    rows=[]
    for line in (ROOT/'COORD/SUBMISSIONS.md').read_text().splitlines():
        fields=[p.strip() for p in line.split('|')]
        if len(fields)<8 or not fields[4].lower().startswith('soumis'):continue
        m=re.search(r'\d+[,.]\d+',fields[5])
        if not m:continue
        checkpoint=fields[3].strip('`')
        if not checkpoint.endswith('.pt'):continue
        path=(ROOT/checkpoint).resolve()
        if not path.is_relative_to(ROOT/'orehn/ckpt') and not path.is_relative_to(HERE/'ckpt'):continue
        rows.append({'score':float(m.group().replace(',','.')),'checkpoint':checkpoint,'time':fields[1],'who':fields[2]})
    return sorted(rows,key=lambda r:r['score'],reverse=True)


def finished_source(checkpoint):
    path=ROOT/checkpoint
    info=ROOT/'orehn/logs'/f'{path.stem}.json'
    if not path.is_file() or not info.is_file():return False
    data=json.loads(info.read_text())
    return data['final']['step']==data['args']['steps'] and path.stat().st_size<450_000_000


def refresh(state,phase):
    records=scores();best=records[0] if records else None
    reports=[]
    for job in state['jobs']:
        done=HERE/'logs'/f"{job['name']}_done.json"
        gate=HERE/'logs'/f"{job['name']}_gate.json"
        if done.exists():job['status']='finished'
        elif any((QUEUE/f'gpu3/{d}'/job['queue_name']).exists() for d in ['failed']):job['status']='failed'
        elif (QUEUE/'gpu3/running'/job['queue_name']).exists():job['status']='running'
        elif (QUEUE/'gpu3/done'/job['queue_name']).exists():job['status']='finished_without_report'
        if gate.exists():
            data=json.loads(gate.read_text())
            reports.append(f"- `{job['name']}` : retain proxy conservateur {100*data['retention_proxy_conservative']:.1f}% ; "
                           f"retain amélioré={data['retain_improved_locally']}, oubli préservé={data['forget_preserved_locally']}. "
                           'Accuracy non certifiée ; aucune soumission automatique.')
        else:reports.append(f"- `{job['name']}` : {job['status']}.")
    text=f'''{START}
# Résumé de nuit pour Orehn — 7h40

Dernière mise à jour : {now():%d/%m %H:%M:%S %Z}. État : {phase}.

- Meilleur score officiel d'équipe connu : {best['score'] if best else 'inconnu'} ({best['checkpoint'] if best else 'inconnu'}, {best['time'] if best else ''}). Le public ne décide pas des micro-variantes ; delta <0,01 ignoré.
- Soumissions Codex pendant le mode nuit : **aucune**. Avant la nuit : A03 soumis avec accord explicite, score 0,0735.
- Sécurité : contrôle public sans vrais labels ; accord et `accord/(1+KL)` ne prouvent pas une accuracy >=75%. Aucun candidat final certifié à ce stade. Les résultats officiels positifs prouvent le seuil serveur de70% sur ces envois, pas une marge locale de75%.
- Piste Codex : réparation du retain après oubli, ancrée aux prédictions du modèle source sur **7 vues** (propre, bright, flip, crop, décalage, bruit, blur). Deux configurations préfixées par source ; trois sources maximum, aucun balayage du score public.
- GPU3 uniquement, dépôts lorsque libre et file vide. GPU2 réservé aux collègues. **Jamais de sbatch/scancel**, aucun tmux/runner Model Equality touché ; aucun token lu par les expériences.
- Ce qui reste : obtenir une validation labellisée pertinente ou une estimation d'accuracy justifiée avant toute soumission autonome ; revoir les compromis locaux ci-dessous. Les meilleurs scores publics seuls ne suffisent pas.

{chr(10).join(reports) if reports else '- Expériences de réparation en préparation / attente GPU3.'}

Artefacts : `codex/logs/*_gate.json`, `*_done.json`, logits multivues, `COORD/RESULTS.csv`. Résumé de la méthode : `codex/NIGHT_PLAN.md`. Le coordinateur s'arrête au plus tard à {state['until']} ; il ne relance aucun nœud.
{END}
'''
    target=HERE/'STATUS.md'
    prior=target.read_text() if target.exists() else ''
    if START in prior and END in prior:prior=prior.split(END,1)[1].lstrip()
    temp=HERE/'.STATUS.night.tmp';temp.write_text(text+'\n'+prior);os.replace(temp,target)
    state['updated']=now().isoformat();state['phase']=phase;save_json(STATE,state)


def queue_if_free(state):
    status=QUEUE/'STATUS.txt'
    if not status.exists() or time.time()-status.stat().st_mtime>90:return 'Worker sans état récent ; attente, aucune relance'
    if (QUEUE/'STOP').exists():return 'Worker STOP ; attente, aucune relance'
    if not re.search(r'^GPU 3 .*: libre ; 0 en attente$',status.read_text(),re.M):return 'GPU3 occupé ; attente'
    if list((QUEUE/'gpu3').glob('*.sh')):return 'File GPU3 non vide ; attente'
    available=[s for s in scores() if finished_source(s['checkpoint'])]
    if not available:return 'Aucune source officielle terminée exploitable'
    best=available[0]
    if not state['sources']:
        state['sources'].append(best)
    elif best['checkpoint'] not in {s['checkpoint'] for s in state['sources']} and len(state['sources'])<3 and best['score']>=max(s['score'] for s in state['sources'])+.01:
        state['sources'].append(best)
    for source in state['sources']:
        stem=Path(source['checkpoint']).stem
        for cfg in CONFIGS:
            name=f"night_repair_{stem}_{cfg['tag']}"
            if any(j['name']==name for j in state['jobs']):continue
            script=HERE/'jobs'/f'{name}.sh';script.parent.mkdir(exist_ok=True)
            command=[str(ROOT/'.venv/bin/python'),'-u',str(HERE/'repair.py'),'--source',source['checkpoint'],
                     '--name',name,'--last',str(cfg['last']),'--anchor',str(cfg['anchor']),
                     '--steps',str(cfg['steps']),'--lr',str(cfg['lr'])]
            script.write_text('#!/bin/bash\n# TIMEOUT=1800\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+
                              '\nexport OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n'+shlex.join(command)+'\n')
            queue_name=f'90_codex_{name}.sh';target=QUEUE/'gpu3'/queue_name
            # Atomic link: worker cannot open a half-written script; no overwrite.
            os.link(script,target)
            state['jobs'].append({'name':name,'source':source,'config':cfg,'queue_name':queue_name,'status':'queued','time':now().isoformat()})
            with (ROOT/'COORD/BOARD.md').open('a') as f:
                f.write(f"\n- {now():%d/%m %H:%M} — Codex nuit : GPU3 observé libre, dépôt {queue_name}, source={source['checkpoint']}, réparation retain {cfg['tag']}, sorties codex/. Pas de soumission automatique (accuracy non certifiée).\n")
            print('QUEUED',queue_name,flush=True)
            return f'Essai déposé : {name}'
    return 'Essais prédéclarés terminés ou déjà déposés ; surveillance des mises à jour'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--until',default='2026-11-22T07:35:00+01:00');ap.add_argument('--once',action='store_true');a=ap.parse_args()
    import fcntl
    lease=(HERE/'night_watch.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (HERE/'night_watch.pid').write_text(str(os.getpid())+'\n')
    deadline=dt.datetime.fromisoformat(a.until)
    state=json.loads(STATE.read_text()) if STATE.exists() else {'sources':[],'jobs':[],'until':a.until}
    while now()<deadline:
        if (HERE/'NIGHT_STOP').exists():refresh(state,'Arrêt demandé via codex/NIGHT_STOP');return
        try:
            phase=queue_if_free(state);refresh(state,phase)
        except Exception as exc:
            print(now().isoformat(),'WATCH_ERROR',repr(exc),flush=True)
            try:refresh(state,'Blocage du coordinateur : '+repr(exc))
            except OSError:pass
        if a.once:return
        time.sleep(30)
    refresh(state,'Fin de surveillance à 7h35 ; consulter les essais terminés ci-dessous')


if __name__=='__main__':main()
