#!/bin/bash
set -e
cd /scratch/siu856580487/qatt
mkdir -p logs
jid=$(sbatch --parsable run_qintra_dataeff.slurm)
echo "qintra data-eff array job: $jid  (25 tasks: 5 N x 5 seeds)"
jidm=$(sbatch --parsable --dependency=afterok:$jid 99_merge_qintra.slurm)
echo "qintra merge job: $jidm"
echo
echo "watch:  squeue -u \$USER"
echo "result: cat logs/99_merge_qintra_${jidm}.out   (when done)"
