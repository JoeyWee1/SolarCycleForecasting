#!/bin/bash
#SBATCH -J gpr20_mwd
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/zyw26/rds/hpc-work/logs/%j.log

mkdir -p /home/zyw26/rds/hpc-work/gpr20_results

source /home/zyw26/Thesis/ThesisVenv/bin/activate

python /home/zyw26/Thesis/run_gpr20.py \
    --data_dir /home/zyw26/Thesis/Data/mwd \
    --output_dir /home/zyw26/rds/hpc-work/gpr20_results \
    --star_type G \
    --n_windows 5 \
    --skip_existing
