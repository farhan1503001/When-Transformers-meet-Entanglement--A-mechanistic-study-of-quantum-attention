#!/bin/bash
#SBATCH --job-name=qintra_merge
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/99_merge_qintra_%j.out
#SBATCH --error=logs/99_merge_qintra_%j.err
set -e
cd /scratch/siu856580487/qatt
module load python 2>/dev/null || true
source /scratch/siu856580487/qatt/env/bin/activate
export CUDA_VISIBLE_DEVICES=""
python merge_qintra_dataeff.py --in_dir ./agnews_results
mkdir -p ~/qatt_results_backup
cp agnews_results/agnews_Cqintra_*.json agnews_results/qintra_dataeff_agnews_merged.json ~/qatt_results_backup/ 2>/dev/null || true
echo "backed up."
