#!/bin/bash
# Submit the full pipeline. Each stage runs only if the previous succeeded.
set -e
cd /scratch/siu856580487/qatt
mkdir -p logs

jid_setup=$(sbatch --parsable 00_setup_smoke.slurm)
echo "setup  job: $jid_setup"

jid_run=$(sbatch --parsable --dependency=afterok:$jid_setup run_agnews.slurm)
echo "array  job: $jid_run  (waits for setup)"

jid_merge=$(sbatch --parsable --dependency=afterok:$jid_run 99_merge.slurm)
echo "merge  job: $jid_merge (waits for all 5 seeds)"

echo
echo "submitted. watch with:  squeue -u \$USER"
echo "setup log:  tail -f logs/00_setup_${jid_setup}.out"
echo "seed0 log:  tail -f logs/agnews_${jid_run}_0.out"
