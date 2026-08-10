#!/bin/bash
set -e
cd /scratch/siu856580487/qatt
mkdir -p logs
jid=$(sbatch --parsable run_agnews_qubits_qintra.slurm)
echo "qintra qubit-sweep array: $jid  (3 seeds x nq{2,4,8})"
jidm=$(sbatch --parsable --dependency=afterok:$jid 99_merge_qubits.slurm)
echo "merge (all conds incl qintra): $jidm"
echo
echo "watch:  squeue -u \$USER"
echo "result: cat logs/99_merge_nq_${jidm}.out   (when done)"
