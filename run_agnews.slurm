#!/bin/bash
#SBATCH --job-name=qatt_agnews
#SBATCH --partition=debug
#SBATCH --array=0-4                 # one task per seed; all run in parallel
#SBATCH --cpus-per-task=32          # node has 128 cores; 32 threads per seed
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/agnews_%A_%a.out
#SBATCH --error=logs/agnews_%A_%a.err

set -e
cd /scratch/siu856580487/qatt
mkdir -p logs agnews_results
module load python 2>/dev/null || true
source /scratch/siu856580487/qatt/env/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""

# preflight: fail fast (no wasted 24h alloc) if env isn't fixed
python -c "import pennylane as qml; qml.device('default.qubit', wires=2)" 2>/dev/null \
  || { echo "ENV NOT READY — submit 00_setup_smoke.slurm first (autoray pin)."; exit 1; }

echo "host=$(hostname) seed=$SLURM_ARRAY_TASK_ID cpus=$SLURM_CPUS_PER_TASK pwd=$(pwd)"

# AG News in its own setting: 5000 train, max_len 53 (NOT TREC-matched)
python qatt_agnews.py \
    --seed $SLURM_ARRAY_TASK_ID \
    --data_dir ./agnews \
    --out_dir ./agnews_results \
    --limit_train 5000 \
    --max_len 53 \
    --epochs 20 \
    --backend default.qubit \
    --threads $SLURM_CPUS_PER_TASK

# Block-1 depth run: add  --quantum_in_block 1   (writes separate *_block1_* files)
