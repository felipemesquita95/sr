#!/usr/bin/env bash
# Recria as features e executa os experimentos comparáveis de BrSD e VCTK.
set -euo pipefail

cd "$(dirname "$0")/.."

hdd=/media/lsmsqt/HDD
vctk_zip="$hdd/datasets/vctk/VCTK-Corpus-0.92.zip"
vctk_root="$hdd/datasets/vctk/VCTK-Corpus-0.92"
brsd_root="$hdd/datasets/brsd/utterances"

mountpoint -q "$hdd" || { echo "HDD não montado: $hdd" >&2; exit 1; }
[[ -f "$vctk_zip" && -d "$vctk_root/wav48_silence_trimmed" ]] || {
    echo 'Corpus VCTK ausente ou não descompactado.' >&2; exit 1;
}
[[ -d "$brsd_root" ]] || { echo 'Corpus BrSD ausente.' >&2; exit 1; }
for number in $(seq 1 400); do
    [[ -f "$brsd_root/$number.wav" ]] || { echo "BrSD sem $number.wav" >&2; exit 1; }
done
[[ "$(readlink -f runs/features)" == "$hdd/sr_project/features" ]] || {
    echo 'runs/features não aponta para o HDD.' >&2; exit 1;
}
[[ "$(readlink -f runs/models)" == "$hdd/sr_project/models" ]] || {
    echo 'runs/models não aponta para o HDD.' >&2; exit 1;
}

export KERAS_BACKEND=torch
python=.venv/bin/python

"$python" experiments/ingest_vctk.py \
    --config configs/vctk8k.env --zip "$vctk_zip" \
    --features-root runs/features --dry-run
cp "$vctk_root/speaker-info.txt" runs/features/vctk_speaker_info.txt

"$python" experiments/run_experiment.py --config configs/brsd.env --preprocess-only
"$python" experiments/run_experiment.py --config configs/vctk8k.env --preprocess-only
"$python" experiments/run_experiment.py --config configs/vctk8k_mic2.env --preprocess-only
"$python" experiments/report_feature_lengths.py

if [[ "${SKIP_BRSD_TRAIN:-0}" != 1 ]]; then
    "$python" experiments/run_experiment.py --config configs/brsd.env
fi
# Os dois microfones do VCTK usam a mesma menor duração observada entre eles.
vctk_min=$("$python" -c 'import json; p="runs/features/relatorio_comprimentos_8k.json"; print(json.load(open(p))["vctk_combinado"]["menor"]["quadros"])')
MAX_FRAMES_CAP="$vctk_min" "$python" experiments/run_experiment.py --config configs/vctk8k.env
MAX_FRAMES_CAP="$vctk_min" "$python" experiments/run_experiment.py --config configs/vctk8k_mic2.env
