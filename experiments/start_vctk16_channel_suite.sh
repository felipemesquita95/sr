#!/usr/bin/env bash
set -euo pipefail
cd /home/lsmsqt/Documents/sr
mountpoint -q /media/lsmsqt/HDD
systemd-run --user --unit=sr-vctk16-channel-suite --collect \
  --working-directory=/home/lsmsqt/Documents/sr \
  /home/lsmsqt/Documents/sr/.venv/bin/python -u \
  /home/lsmsqt/Documents/sr/experiments/run_vctk16_channel_suite.py
gnome-terminal --title='CNN, temporal e x-vector — CMN e RASTA GPU' -- \
  bash -lc 'journalctl --user -u sr-vctk16-channel-suite.service -f -n 20 -o cat'
