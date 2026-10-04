#!/usr/bin/env bash
set -euo pipefail

cd /home/lsmsqt/Documents/sr
root=output/vctk100_xvector_silero_20260929/results

while true; do
  read -r state completed current < <(python3 - "$root/status.json" <<'PY'
import json
import sys
status = json.load(open(sys.argv[1]))
print(status['status'], status.get('models_completed', 0),
      status.get('current_model') or '-')
PY
  )
  clear
  printf 'X-vector VCTK 100 | %s/20 concluídos | faltam %s\n' "$completed" "$((20-completed))"
  printf 'Estado: %s | modelo atual: %s\n\n' "$state" "$current"
  if [[ "$current" != '-' && -f "$root/models/$current/execution.log" ]]; then
    tail -n 18 "$root/models/$current/execution.log"
  else
    tail -n 15 "$root/suite_execution.log"
  fi
  if [[ "$state" == complete || "$state" == failed ]]; then
    printf '\nExecução encerrada. Pressione Enter para fechar.\n'
    read -r
    exit 0
  fi
  sleep 3
done
