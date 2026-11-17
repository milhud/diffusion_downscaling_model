#!/usr/bin/bash
#SBATCH -J era5_dl
#SBATCH --partition=datamove
#SBATCH --account=s1001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=1980-2020%6
#SBATCH -o /discover/nobackup/hpmille1/data/era5_data/logs/era5_%a.out
#SBATCH -e /discover/nobackup/hpmille1/data/era5_data/logs/era5_%a.err

module load python/GEOSpyD/24.3.0-0/3.12
/discover/nobackup/hpmille1/download_pipeline/venv/bin/python \
  /discover/nobackup/hpmille1/diffusion_downscaling_model/scripts/download_era5_arco.py "$SLURM_ARRAY_TASK_ID"
