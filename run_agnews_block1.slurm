#!/bin/bash
#SBATCH --job-name=qatt_agnews_b1
#SBATCH --partition=debug
#SBATCH --array=0-4                 # one task per seed; all run in parallel
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/agnews_b1_%A_%a.out
#SBATCH --error=logs/agnews_b1_%A_%a.err

set -e
cd /scratch/siu856580487/qatt
mkdir -p logs agnews_results
module load python 2>/dev/null || true
source /scratch/siu856580487/qatt/env/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""

# preflight: fail fast if env isn't ready
python -c "import pennylane as qml; qml.device('default.qubit', wires=2)" 2>/dev/null \
  || { echo "ENV NOT READY — run 00_setup_smoke.slurm first (autoray pin)."; exit 1; }

echo "host=$(hostname) seed=$SLURM_ARRAY_TASK_ID cpus=$SLURM_CPUS_PER_TASK pwd=$(pwd)"

# SAME setting as block 0 (5000 train, max_len 53) but quantum head in the LAST block.
# Writes agnews_block1_seed<N>.json — separate files, cannot touch block-0 results.
python qatt_agnews.py \
    --seed $SLURM_ARRAY_TASK_ID \
    --data_dir ./agnews \
    --out_dir ./agnews_results \
    --limit_train 5000 \
    --max_len 53 \
    --epochs 20 \
    --quantum_in_block 1 \
    --backend default.qubit \
    --threads $SLURM_CPUS_PER_TASK
