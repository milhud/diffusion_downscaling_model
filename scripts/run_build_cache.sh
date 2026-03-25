#!/usr/bin/bash
#SBATCH -J build_cache
#SBATCH --partition=columbia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --qos=columbia
#SBATCH -o cache_output.%j
#SBATCH -e cache_error.%j
#SBATCH --account=columbia

# ──────────────────────────────────────────────────────────────────────
# Build regridded cache for all variables.
#
# Usage:
#   sbatch scripts/run_build_cache.sh /path/to/output/cache
#   sbatch scripts/run_build_cache.sh /path/to/cache --years 1980 1981 1982
#
# Time estimate: ~10-15 min per year, ~7-10 hours for all 41 years.
# Size estimate: ~5 GB/year for 6+6 variables = ~200 GB total.
# ──────────────────────────────────────────────────────────────────────

cd /mnt/home/hmiller/diffusion_downscaling_model

module purge
module load Python/3.10.15
source ~/venv/bin/activate

OUTPUT_DIR="${1:?Usage: sbatch run_build_cache.sh /path/to/output/cache}"
shift

python -u scripts/build_cache.py --output_dir "$OUTPUT_DIR" --data_dir data "$@"

exit 0
