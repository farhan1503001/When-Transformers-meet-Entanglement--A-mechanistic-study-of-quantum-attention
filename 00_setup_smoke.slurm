#!/bin/bash
#SBATCH --job-name=qatt_setup
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/00_setup_%j.out
#SBATCH --error=logs/00_setup_%j.err

set -e
cd /scratch/siu856580487/qatt
mkdir -p logs smoke
module load python 2>/dev/null || true
source /scratch/siu856580487/qatt/env/bin/activate
export CUDA_VISIBLE_DEVICES=""

echo "=== [1/3] pinning autoray (fixes PennyLane 0.38 import) ==="
pip install "autoray==0.7.1" 2>&1 | tail -3 || {
  echo "PIP FAILED — compute node may have no internet."
  echo "Run this ONCE on the login node, then resubmit: pip install autoray==0.7.1"
  exit 1; }

echo "=== [2/3] import check ==="
python -c "import pennylane as qml; print('pennylane', qml.version(), 'OK'); qml.device('default.qubit', wires=2); print('device OK')"

echo "=== [3/3] tiny end-to-end smoke (200 ex, max_len 12, 1 epoch) ==="
python qatt_agnews.py --seed 0 --data_dir ./agnews --out_dir ./smoke \
    --limit_train 200 --max_len 12 --epochs 1
test -f ./smoke/agnews_seed0.json && echo "SMOKE OK: smoke/agnews_seed0.json written" \
    || { echo "SMOKE FAILED: no output file"; exit 1; }
echo "=== SETUP COMPLETE ==="
