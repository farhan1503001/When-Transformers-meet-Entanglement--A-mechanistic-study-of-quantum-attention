#!/bin/bash
set -e
cd /scratch/siu856580487/qatt
mkdir -p logs
jid_run=$(sbatch --parsable run_agnews_block1.slurm)
echo "block1 array job: $jid_run"
jid_merge=$(sbatch --parsable --dependency=afterok:$jid_run 99_merge_block1.slurm)
echo "block1 merge job: $jid_merge (waits for all 5 seeds)"
echo
echo "watch:  squeue -u \$USER"
echo "seed0:  tail -f logs/agnews_b1_${jid_run}_0.out"
echo "result: cat logs/99_merge_b1_${jid_merge}.out   (when done)"
