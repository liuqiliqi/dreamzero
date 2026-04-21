#!/bin/bash
###############################################################################
# Slurm job: Run IsaacLab simulation evaluation (connects to DreamZero server)
#
# Prerequisites:
#   The DreamZero policy server must already be running on a node.
#   Submit it first:  sbatch scripts/slurm/serve_dreamzero.slurm
#   Check which node:  squeue -u $USER  (e.g. "blue-space")
#
# Usage:
#   SERVER_NODE=blue-space sbatch scripts/slurm/run_eval.slurm
###############################################################################

#SBATCH --job-name=dreamzero-eval
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%j.log
# NOTE: Eval MUST run on a DIFFERENT node from the server to avoid GPU conflicts.

# ── Configuration ─────────────────────────────────────────────────────────────
SERVER_NODE="${SERVER_NODE:-blue-space}"
SERVER_PORT="${SERVER_PORT:-8000}"
EPISODES="${EPISODES:-10}"
SCENE="${SCENE:-1}"

PROJECT_DIR="/fact_home/qiliu/worldmodel/sim-evals"

# ── Setup ─────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"
source .venv/bin/activate

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Server:        $SERVER_NODE:$SERVER_PORT"
echo "Episodes:      $EPISODES"
echo "Scene:         $SCENE"
echo "Python:        $(python --version)"
echo "============================================"

# ── Launch ────────────────────────────────────────────────────────────────────
# Export server info so the wrapper script can patch the client
export SERVER_NODE
export SERVER_PORT

# Omniverse/IsaacLab needs to enumerate GPUs itself; Slurm's CUDA_VISIBLE_DEVICES
# puts CUDA in a state that Omniverse interprets as "bad". Unset it and let
# Omniverse pick the GPU via its own Vulkan/CUDA discovery.
unset CUDA_VISIBLE_DEVICES

# Force Omniverse to use GPU 0 and enable headless Vulkan rendering
export OMNI_KIT_ALLOW_NO_DISPLAY=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

python /fact_home/qiliu/worldmodel/dreamzero/scripts/slurm/run_eval_wrapper.py \
    --episodes "$EPISODES" \
    --headless \
    --scene "$SCENE"

echo "Evaluation complete!"
