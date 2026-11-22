"""CPU-only helper: deposit ONE Codex script only after GPU 2/3 is free.

No scheduler operations, no process control, no tokens, no teammate mutations.
Run attached, with a finite deadline. The shared worker starts the GPU work.
"""
from pathlib import Path
import argparse
import datetime
import os
import re
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--script',default='run_followup.sh')
    ap.add_argument('--name',default='85_codex_followup.sh')
    ap.add_argument('--wait-minutes',type=float,default=20)
    a=ap.parse_args()
    here=Path(__file__).resolve().parent
    queue=here.parents[1]/'gpu_queue'
    source=here/a.script
    assert source.parent==here and source.is_file()
    assert a.name.startswith('85_codex_') and Path(a.name).name==a.name and a.name.endswith('.sh')
    limit=time.monotonic()+60*a.wait_minutes
    while time.monotonic()<limit:
        status=queue/'STATUS.txt'
        if status.exists() and time.time()-status.stat().st_mtime<40 and not (queue/'STOP').exists():
            text=status.read_text()
            for gpu in [2,3]:
                folder=queue/f'gpu{gpu}'
                available=re.search(rf'^GPU {gpu} .*: libre ; 0 en attente$',text,re.M)
                if not available or list(folder.glob('*.sh')): continue
                target=folder/a.name
                # Exclusive creation avoids overwriting any existing file.
                with target.open('x') as handle: handle.write(source.read_text())
                when=datetime.datetime.now().strftime('%d/%m %H:%M')
                message=f'{when} — Codex : GPU {gpu} observe libre (STATUS frais, aucune attente), depot {a.name}.\n'
                with (here/'STATUS.md').open('a') as handle: handle.write('\n'+message)
                with (here.parent/'COORD/BOARD.md').open('a') as handle: handle.write('\n- '+message)
                print('QUEUED',str(target),flush=True)
                return
        time.sleep(15)
    print('NO DEPOSIT: GPUs never became free before finite deadline',flush=True)
    raise SystemExit(2)


if __name__=='__main__': main()
