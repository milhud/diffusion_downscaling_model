#!/usr/bin/bash
#SBATCH -J preprocess_%a
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --qos=columbia
#SBATCH -o preprocess_output.%A_%a
#SBATCH -e preprocess_error.%A_%a
#SBATCH --account=columbia
#SBATCH --array=0-40  # 41 years: 1980-2020

# ──────────────────────────────────────────────────────────────────────
# Parallel preprocessing: one SLURM array task per year.
# Each task regrids ERA5 and caches data for one year (~5-10 min).
#
# Usage:
#   sbatch run_preprocess.sh          # all 41 years in parallel
#   sbatch --array=0-4 run_preprocess.sh  # just 1980-1984
# ──────────────────────────────────────────────────────────────────────

cd /mnt/home/hmiller/diffusion_downscaling_model

module purge
module load Python/3.10.15
source ~/venv/bin/activate

YEAR=$((1980 + SLURM_ARRAY_TASK_ID))
echo "Processing year: $YEAR (task $SLURM_ARRAY_TASK_ID)"

python preprocess_cache.py --year $YEAR --data_dir data --cache_dir cached_data

exit 0
