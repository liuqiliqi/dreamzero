#!/bin/bash
###############################################################################
# Slurm job: Launch the DreamZero inference (WebSocket policy server)
#
# This cluster has two GPU types:
#   - "ai" partition:  NVIDIA H20 x8 per node (1.2 TB RAM)  -- best for large models
#   - "dev" partition: NVIDIA L20 x4 per node (600 GB RAM)   -- default, 12h limit
#
# Usage:
#   # Single GPU on "ai" partition (H20):
#   sbatch scripts/slurm/serve_dreamzero.slurm
#
#   # 4 GPUs with tensor parallelism:
#   sbatch --gpus=4 scripts/slurm/serve_dreamzero.slurm
#
#   # Use "dev" partition (L20) instead:
#   sbatch --partition=dev --gres=gpu:nvidia_l20:1 scripts/slurm/serve_dreamzero.slurm
#
# After submission:
#   squeue -u $USER          # see your job status and which node it's on
#   cat logs/serve_<jobid>.log   # check server output
#   scancel <jobid>          # cancel the job when done
###############################################################################

#SBATCH --job-name=dreamzero-serve
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/serve_%j.log

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/fact_data/qiliu/dreamzero_weights/DreamZero-DROID}"
PORT="${PORT:-8000}"
PROJECT_DIR="/fact_home/qiliu/worldmodel/dreamzero"

# ── Setup ─────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"

# Activate the project's virtual environment
source .venv/bin/activate

# Print useful info for debugging
echo "============================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "GPUs:         $SLURM_GPUS_ON_NODE"
echo "Python:       $(python --version)"
echo "Model path:   $MODEL_PATH"
echo "Port:         $PORT"
echo "============================================"
echo ""
echo ">>> To connect from another terminal, run:"
echo ">>>   python eval_utils/run_sim_eval.py --host $SLURM_NODELIST --port $PORT"
echo ""

# ── Count allocated GPUs ─────────────────────────────────────────────────────
# SLURM_GPUS_ON_NODE is set by Slurm when using --gres or --gpus
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"

# ── Launch ────────────────────────────────────────────────────────────────────
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Starting with $NUM_GPUS GPUs (tensor parallelism)..."
    python -m torch.distributed.run \
        --standalone --nproc_per_node="$NUM_GPUS" \
        eval_utils/serve_dreamzero.py \
        --model-path "$MODEL_PATH" \
        --port "$PORT"
else
    echo "Starting with 1 GPU..."
    python eval_utils/serve_dreamzero.py \
        --model-path "$MODEL_PATH" \
        --port "$PORT"
fi
