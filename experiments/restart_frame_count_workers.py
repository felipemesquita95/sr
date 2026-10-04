#!/usr/bin/env python3
"""Restart only this audit's worker pool, retaining its completed count rows."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
result=Path(sys.argv[1]).resolve()
script=root/'experiments/audit_frame_availability.py'
matched=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        args=(entry/'cmdline').read_bytes().split(b'\0')
        if not any(a.endswith(b'audit_frame_availability.py') for a in args):
            continue
        if b'--output' not in args:
            continue
        target=args[args.index(b'--output')+1].decode()
        if (root/target).resolve()!=result:
            continue
        pid=int(entry.name)
        children=[int(n) for n in (entry/f'task/{pid}/children').read_text().split()]
        matched.append((pid,children))
    except (FileNotFoundError,ProcessLookupError,PermissionError):
        continue
for pid,children in matched:
    for target in [pid]+children:
        try:
            os.kill(target,signal.SIGTERM)
        except ProcessLookupError:
            pass
time.sleep(1)
data=result/'segments_counts.jsonl'
lines=data.read_text().splitlines()
assert all(json.loads(line) for line in lines), 'Incomplete count row; stop rather than change it'
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
log=result/'scan_workers8.log'
with log.open('x') as output:
    child=subprocess.Popen([str(root/'.venv/bin/python'),str(script),'--output',str(result),'--workers','8'],
        cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
        start_new_session=True,close_fds=True)
(result/'execution_workers8.json').write_text(json.dumps(dict(pid=child.pid,workers=8,
    resumed_completed_rows=len(lines),stopped_audit_processes=matched),indent=2)+'\n')
print(f'Audit restarted with 8 workers, PID {child.pid}, retained {len(lines)} completed rows.')
