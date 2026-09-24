#!/usr/bin/bash
#SBATCH -J cache_part2
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --constraint=rome
#SBATCH --time=12:00:00
#SBATCH --qos=alla100
#SBATCH -o cache_p2_output.%j
#SBATCH -e cache_p2_error.%j
#SBATCH --account=s1001

cd /gpfsm/dnb33/hpmille1/diffusion_downscaling_model
module purge
module load python/GEOSpyD/24.3.0-0/3.12

python -u scripts/build_cache.py --output_dir /discover/nobackup/sduan/.data \
    --data_dir data --years 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020

exit 0
