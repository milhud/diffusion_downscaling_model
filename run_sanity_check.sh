#!/usr/bin/bash
#SBATCH -J sanity_check
#SBATCH --partition=columbia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=6:00:00
#SBATCH --qos=columbia
#SBATCH -o sanity_check_output.%j
#SBATCH -e sanity_check_error.%j
#SBATCH --account=columbia

cd /mnt/home/hmiller/diffusion_downscaling_model

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
module load Python/3.10.15
source ~/venv/bin/activate

# Pass through any extra args, e.g.: sbatch run_sanity_check.sh --extensive
python sanity_check.py --device cuda --plot_dir sanity_plots "$@"

exit 0
