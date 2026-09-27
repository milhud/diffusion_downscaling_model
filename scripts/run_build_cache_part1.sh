#!/usr/bin/bash
#SBATCH -J cache_part1
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --constraint=rome
#SBATCH --time=12:00:00
#SBATCH --qos=alla100
#SBATCH -o cache_p1_output.%j
#SBATCH -e cache_p1_error.%j
#SBATCH --account=s1001

cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model
module purge
module load python/GEOSpyD/24.3.0-0/3.12

python -u scripts/build_cache.py --output_dir /discover/nobackup/sduan/.data \
    --data_dir data --years 1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000

exit 0
