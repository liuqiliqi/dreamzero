#!/bin/bash
###############################################################################
# Slurm job: DreamZero DROID LoRA training
#
# Usage:
#   sbatch scripts/slurm/train_droid_lora.slurm
#
# After submission:
#   squeue -u $USER              # check job status
#   tail -f logs/train_<jobid>.log   # watch training progress
#   scancel <jobid>              # cancel if needed
###############################################################################

#SBATCH --job-name=dreamzero-train
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h20:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.log

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_DIR="/fact_home/qiliu/worldmodel/dreamzero"

# ── Setup ─────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"
source .venv/bin/activate

echo "============================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "GPUs:     $SLURM_GPUS_ON_NODE"
echo "Python:   $(python --version)"
echo "============================================"

# ── Launch training ───────────────────────────────────────────────────────────
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-8}"
bash scripts/train/droid_training_lora.sh
