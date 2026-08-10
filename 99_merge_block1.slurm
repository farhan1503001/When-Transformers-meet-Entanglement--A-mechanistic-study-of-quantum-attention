#!/bin/bash
#SBATCH --job-name=qatt_merge_b1
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/99_merge_b1_%j.out
#SBATCH --error=logs/99_merge_b1_%j.err

set -e
cd /scratch/siu856580487/qatt
module load python 2>/dev/null || true
source /scratch/siu856580487/qatt/env/bin/activate
export CUDA_VISIBLE_DEVICES=""

echo "=== merging block-1 (depth) results ==="
python merge_agnews.py --in_dir ./agnews_results --block1

echo "=== backing up to /home ==="
mkdir -p ~/qatt_results_backup
cp agnews_results/*block1*.json ~/qatt_results_backup/
ls -la ~/qatt_results_backup/
