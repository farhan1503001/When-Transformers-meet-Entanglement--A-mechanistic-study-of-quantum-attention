#!/bin/bash
# Fills the missing qintra condition in the AG News qubit-width sweep (Exp D),
# matched to the existing sweep: limit_train 1500, max_len 20, epochs 12, 3 seeds,
# nq in {2,4,8}. Writes agnews_D_qintra_nq<NQ>_seed<S>.json (separate files;
# existing D_nq* files are never touched).
#SBATCH --job-name=qintra_nq
#SBATCH --partition=debug
#SBATCH --array=0-2                 # 3 seeds (mirrors the existing Exp D sweep)
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/qintra_nq_%A_%a.out
#SBATCH --error=logs/qintra_nq_%A_%a.err

set -e
cd /scratch/siu856580487/qatt
mkdir -p logs agnews_results
module load python 2>/dev/null || true
source /scratch/siu856580487/qatt/env/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""

python -c "import pennylane as qml; qml.device('default.qubit', wires=2)" 2>/dev/null \
  || { echo "ENV NOT READY — run 00_setup_smoke.slurm first."; exit 1; }

S=$SLURM_ARRAY_TASK_ID
COMMON="--seed $S --data_dir ./agnews --out_dir ./agnews_results \
        --limit_train 1500 --max_len 20 --epochs 12 \
        --backend default.qubit --threads $SLURM_CPUS_PER_TASK"

# cheap widths first, checkpointed; nq=8 last
for NQ in 2 4 8; do
    echo "=== seed $S  qintra  n_qubits=$NQ ==="
    python qatt_agnews.py $COMMON --conditions qintra --tag D_qintra_nq${NQ} --n_qubits $NQ
done
echo "[done] qintra qubit sweep seed $S"
