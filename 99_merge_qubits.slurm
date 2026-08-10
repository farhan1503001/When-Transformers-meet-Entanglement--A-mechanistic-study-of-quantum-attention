#!/bin/bash
#SBATCH --job-name=qatt_merge_nq
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/99_merge_nq_%j.out
#SBATCH --error=logs/99_merge_nq_%j.err
set -e
cd /scratch/siu856580487/qatt
module load python 2>/dev/null || true
source /scratch/siu856580487/qatt/env/bin/activate
export CUDA_VISIBLE_DEVICES=""
python merge_qubits.py --in_dir ./agnews_results
mkdir -p ~/qatt_results_backup
cp agnews_results/agnews_D_*.json agnews_results/agnews_qubitsweep_summary.json ~/qatt_results_backup/ 2>/dev/null || true
echo "backed up."
