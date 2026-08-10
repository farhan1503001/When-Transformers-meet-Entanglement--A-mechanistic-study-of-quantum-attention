#!/bin/bash
#SBATCH --job-name=qatt_lrctrl
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-74
#SBATCH --output=/scratch/siu856580487/qatt/logs/lrctrl_%A_%a.out
#SBATCH --error=/scratch/siu856580487/qatt/logs/lrctrl_%A_%a.err

# 75 runs = 5 conditions x 3 matched LRs x 5 seeds, indexed by SLURM_ARRAY_TASK_ID.
# AG News, primary config (4 heads, block 0), identical to the primary
# ablation in qatt_agnews.py except both parameter groups share one rate.

set -euo pipefail

SCRATCH=/scratch/siu856580487/qatt
OUTDIR=$SCRATCH/results/lr
mkdir -p "$OUTDIR" "$SCRATCH/logs"

cd "$SCRATCH"
source "$SCRATCH/env/bin/activate"

CONDITIONS=(classical qnone qintra qcross qfull)
LRS=(5e-4 1e-3 2e-3)
SEEDS=(0 1 2 3 4)

N_SEEDS=${#SEEDS[@]}       # 5
N_LRS=${#LRS[@]}           # 3
PER_COND=$(( N_SEEDS * N_LRS ))   # 15 runs per condition

idx=$SLURM_ARRAY_TASK_ID
cond_i=$(( idx / PER_COND ))
rem=$(( idx % PER_COND ))
lr_i=$(( rem / N_SEEDS ))
seed_i=$(( rem % N_SEEDS ))

COND=${CONDITIONS[$cond_i]}
LR=${LRS[$lr_i]}
SEED=${SEEDS[$seed_i]}

echo "task $idx -> condition=$COND lr=$LR seed=$SEED"

python lr_control.py \
    --condition "$COND" \
    --seed "$SEED" \
    --lr "$LR" \
    --out "$OUTDIR" \
    --data_dir "$SCRATCH/agnews"

echo "done task $idx"
