#!/bin/bash
set -e
cd /scratch/siu856580487/qatt
mkdir -p logs
jid=$(sbatch --parsable run_agnews_qubits.slurm)
echo "qubit-sweep array job: $jid"
jidm=$(sbatch --parsable --dependency=afterok:$jid 99_merge_qubits.slurm)
echo "qubit-sweep merge job: $jidm (waits for all seeds)"
echo
echo "watch:  squeue -u \$USER"
echo "seed0:  tail -f logs/agnews_nq_${jid}_0.out"
echo "result: cat logs/99_merge_nq_${jidm}.out   (when done)"
