#!/usr/bin/env python3
"""Detach the audit window from the command session, without creating services."""
import os
from pathlib import Path
import subprocess
import sys
import json
import signal
from datetime import datetime

root=Path(sys.argv[1]).resolve()
script=Path(__file__).with_name('view_silero_min_silence.py').resolve()
state=root/'viewer_state.json'
if state.is_file():
    previous=json.loads(state.read_text())
    pid=previous.get('pid')
    if pid and Path(f'/proc/{pid}/cmdline').exists():
        command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if str(script).encode() in command and str(root).encode() in command:
            os.kill(pid,signal.SIGTERM)
env=dict(os.environ,QT_QPA_PLATFORM='wayland')
log=root/f'viewer_{datetime.now():%Y%m%d_%H%M%S_%f}.log'
with log.open('x') as stream:
    child=subprocess.Popen([sys.executable,str(script),str(root)],env=env,
        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
        start_new_session=True,close_fds=True)
print(f'Viewer PID: {child.pid}\nLog: {log}',flush=True)
