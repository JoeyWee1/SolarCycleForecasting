#!/bin/bash
#SBATCH -J gpr24b_sim
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zyw26/rds/hpc-work/logs/gpr24b_%j.log
#SBATCH --array=0-9   # one job per sampling-rate folder; comment out to run all in one job

# ── sampling-rate folders in the same order as np.linspace(1,90,10) ──────────
RATES=(1.0d 10.9d 20.8d 30.7d 40.6d 50.4d 60.3d 70.2d 80.1d 90.0d)

# When running as a SLURM array pick the folder for this task;
# when running interactively (no SLURM_ARRAY_TASK_ID) run all folders.
if [ -n "${SLURM_ARRAY_TASK_ID+x}" ]; then
    RATE_ARG="--rate ${RATES[$SLURM_ARRAY_TASK_ID]}"
else
    RATE_ARG=""
fi

# ── paths (edit to match your HPC layout) ────────────────────────────────────
REPO=/home/zyw26/Thesis/SolarCycleForecasting
SIM_ROOT=$REPO/Data/simulated
OUT_DIR=/home/zyw26/rds/hpc-work/gpr24b_results

mkdir -p "$OUT_DIR"
mkdir -p /home/zyw26/rds/hpc-work/logs

source /home/zyw26/Thesis/ThesisVenv/bin/activate

# ── adjustable parameters ─────────────────────────────────────────────────────
N_WINDOWS=25        # number of lookahead windows per star
# N_STARS=3         # uncomment to limit stars (useful for testing)
# STAR_TYPE=G       # uncomment to run only one spectral type

python "$REPO/run_gpr24b.py" \
    --sim_root   "$SIM_ROOT"  \
    --out_dir    "$OUT_DIR"   \
    --n_windows  $N_WINDOWS   \
    --skip_existing           \
    ${RATE_ARG}               \
    ${N_STARS:+--n_stars $N_STARS}       \
    ${STAR_TYPE:+--star_type_filter $STAR_TYPE}
