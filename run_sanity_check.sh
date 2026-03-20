#!/usr/bin/bash
#SBATCH -J sanity_check
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=rome
#SBATCH --time=1:00:00
#SBATCH -o sanity_check_output.%j
#SBATCH -e sanity_check_error.%j
#SBATCH --account=s1001

cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model

# Archive old log files (exclude current job's files)
mkdir -p sbatch_logs
rm -rf sanity_plots/*
for f in sanity_check_output.* sanity_check_error.*; do
    case "$f" in
        *."$SLURM_JOB_ID") ;;  # skip current job
        *) mv "$f" sbatch_logs/ 2>/dev/null ;;
    esac
done

module purge
module load python/GEOSpyD/24.3.0-0/3.12

# Pass through any extra args, e.g.: sbatch run_sanity_check.sh --extensive
python sanity_check.py --device cuda --plot_dir sanity_plots "$@"

exit 0
