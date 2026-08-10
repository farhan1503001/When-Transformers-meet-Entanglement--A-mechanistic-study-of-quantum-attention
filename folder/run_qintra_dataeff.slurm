#!/bin/bash
# Fills the missing qintra condition for AG News data-efficiency (Exp C),
# matched EXACTLY to the Colab settings (max_len 53, epochs 20, 5 seeds,
# N in {250,500,1000,2000,5000}, same split seed / vocab / hyperparams).
# One (N, seed) per array task -> 5 N x 5 seeds = 25 tasks.
#SBATCH --job-name=qintra_ceff
#SBATCH --partition=debug
#SBATCH --array=0-24
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/qintra_ceff_%A_%a.out
#SBATCH --error=logs/qintra_ceff_%A_%a.err

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

# decode array index -> (N, seed)
N_LIST=(250 500 1000 2000 5000)
i=$SLURM_ARRAY_TASK_ID
NIDX=$(( i / 5 )); SEED=$(( i % 5 ))
N=${N_LIST[$NIDX]}

echo "task=$i  N=$N  seed=$SEED  (AG News qintra, max_len=53, epochs=20)"

# matches Colab Exp C exactly; only --conditions qintra differs from the 4 already done
python qatt_agnews.py \
    --seed $SEED \
    --data_dir ./agnews \
    --out_dir ./agnews_results \
    --conditions qintra \
    --limit_train $N \
    --max_len 53 \
    --epochs 20 \
    --n_qubits 4 \
    --tag Cqintra_N${N} \
    --backend default.qubit \
    --threads $SLURM_CPUS_PER_TASK

echo "[done] N=$N seed=$SEED"
