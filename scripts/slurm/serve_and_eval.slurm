#!/bin/bash
###############################################################################
# Slurm job: Run DreamZero server + IsaacLab eval on the SAME node
#
# Requests 2 GPUs: one for the model server, one for IsaacLab rendering.
#
# Usage:
#   sbatch scripts/slurm/serve_and_eval.slurm
###############################################################################

#SBATCH --job-name=dreamzero-full
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h20:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/full_%j.log

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/fact_data/qiliu/dreamzero_weights/DreamZero-DROID}"
PORT="${PORT:-8000}"
EPISODES="${EPISODES:-10}"
SCENE="${SCENE:-1}"

DREAMZERO_DIR="/fact_home/qiliu/worldmodel/dreamzero"
SIMEVAL_DIR="/fact_home/qiliu/worldmodel/sim-evals"

echo "============================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "GPUs:         $SLURM_GPUS_ON_NODE"
echo "Model path:   $MODEL_PATH"
echo "Episodes:     $EPISODES"
echo "Scene:        $SCENE"
echo "============================================"

# ── Step 1: Start the DreamZero server in the background (GPU 0) ─────────────
echo ""
echo ">>> Starting DreamZero server on GPU 0..."
cd "$DREAMZERO_DIR"
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python eval_utils/serve_dreamzero.py \
    --model-path "$MODEL_PATH" \
    --port "$PORT" &
SERVER_PID=$!

# Wait for the server to be ready (check if port is listening)
echo ">>> Waiting for server to start (PID=$SERVER_PID)..."
for i in $(seq 1 120); do
    if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
        echo ">>> Server is ready on port $PORT!"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ">>> ERROR: Server process died. Check log above."
        exit 1
    fi
    sleep 5
done

# Final check
if ! ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    echo ">>> ERROR: Server did not start within 10 minutes."
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# ── Step 2: Run the IsaacLab eval (GPU 1) ────────────────────────────────────
echo ""
echo ">>> Starting IsaacLab evaluation on GPU 1..."
cd "$SIMEVAL_DIR"
source .venv/bin/activate

# Let Omniverse discover GPUs itself, but hint it to use GPU 1
unset CUDA_VISIBLE_DEVICES
export OMNI_KIT_ALLOW_NO_DISPLAY=1
export SERVER_NODE=localhost
export SERVER_PORT="$PORT"

python "$DREAMZERO_DIR/scripts/slurm/run_eval_wrapper.py" \
    --episodes "$EPISODES" \
    --headless \
    --scene "$SCENE"

EVAL_EXIT=$?

# ── Step 3: Clean up ─────────────────────────────────────────────────────────
echo ""
echo ">>> Evaluation finished (exit code: $EVAL_EXIT). Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo ">>> Done!"
