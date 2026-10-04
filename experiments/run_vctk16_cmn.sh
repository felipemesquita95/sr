#!/usr/bin/env bash
set -euo pipefail

cd /home/lsmsqt/Documents/sr
unset HSA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export KERAS_BACKEND=torch
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/home/lsmsqt/Documents/sr/tmp/matplotlib

mountpoint -q /media/lsmsqt/HDD
while systemctl --user is-active --quiet sr-vctk16-cmn-gpu-pilot.service; do
  sleep 5
done
.venv/bin/python -u -c 'import torch; assert torch.cuda.is_available(), "GPU unavailable"; x=torch.ones(1, device="cuda"); print("GPU ready:", torch.cuda.get_device_name(0), x.item())'

results=/media/lsmsqt/HDD/sr_project/cmn_vctk16_gpu_b128
baseline=/media/lsmsqt/HDD/sr_project/baseline_vctk16_gpu_b128
.venv/bin/python -u experiments/run_vctk16_xvector.py \
  --mode dynamic --cmn-alpha 1 --batch-size 128 --output "$results"
.venv/bin/python experiments/summarize_vctk16_xvector.py \
  --mode dynamic --root "$results" --cohort output/vctk16_cohort_120.json
.venv/bin/python -u experiments/run_vctk16_xvector.py \
  --mode dynamic --cmn-alpha 0 --batch-size 128 --output "$baseline"
.venv/bin/python experiments/summarize_vctk16_xvector.py \
  --mode dynamic --root "$baseline" --cohort output/vctk16_cohort_120.json
.venv/bin/python experiments/report_vctk16_cmn.py \
  --cmn "$results" --baseline "$baseline"
