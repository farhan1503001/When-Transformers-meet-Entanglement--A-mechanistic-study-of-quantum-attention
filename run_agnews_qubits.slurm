#!/bin/bash
#SBATCH --job-name=qatt_agnews_nq
#SBATCH --partition=debug
#SBATCH --array=0-2                 # 3 seeds for the expensive sweep (mirrors Experiment D)
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/agnews_nq_%A_%a.out
#SBATCH --error=logs/agnews_nq_%A_%a.err

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
echo "host=$(hostname) seed=$S cpus=$SLURM_CPUS_PER_TASK"

# Encoding-bottleneck sweep. Reduced settings so nq=8 (2^8 statevector) is tractable,
# exactly as Experiment D was run on TREC. h = n_qubits//2 is the bottleneck width:
# nq=2 -> h=1 (severe), nq=4 -> h=2 (baseline), nq=8 -> h=4 (loosest).
COMMON="--seed $S --data_dir ./agnews --out_dir ./agnews_results \
        --limit_train 1500 --max_len 20 --epochs 12 \
        --backend default.qubit --threads $SLURM_CPUS_PER_TASK"

# classical reference once (independent of n_qubits)
python qatt_agnews.py $COMMON --conditions classical --tag D_classical --n_qubits 4

# cheap widths first, checkpointed; nq=8 last so you can read nq2/nq4 even if you cancel
for NQ in 2 4 8; do
    echo "=== seed $S  n_qubits=$NQ ==="
    python qatt_agnews.py $COMMON --conditions qnone,qcross,qfull --tag D_nq${NQ} --n_qubits $NQ
done
echo "[done] qubit sweep seed $S"
