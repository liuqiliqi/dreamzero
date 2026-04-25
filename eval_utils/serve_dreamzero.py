"""
Launch the DreamZero policy WebSocket server for Isaac Lab sim evaluation.

Usage (single-GPU, no TP):
    cd /fact_home/qiliu/worldmodel/dreamzero
    CUDA_VISIBLE_DEVICES=4 uv run python eval_utils/serve_dreamzero.py \
        --model-path /fact_data/qiliu/dreamzero_weights/DreamZero-DROID \
        --port 8000

Usage (multi-GPU with Tensor Parallelism):
    cd /fact_home/qiliu/worldmodel/dreamzero
    CUDA_VISIBLE_DEVICES=4,5,6,7 uv run python -m torch.distributed.run \
        --standalone --nproc_per_node=4 \
        eval_utils/serve_dreamzero.py \
        --model-path /fact_data/qiliu/dreamzero_weights/DreamZero-DROID \
        --port 8000

Then in another terminal, run the sim-evals evaluation:
    cd /fact_home/qiliu/worldmodel/sim-evals
    uv run python run_eval.py --host localhost --port 8000 --episodes 10 --headless

Observation key mapping (client → model):
    observation/exterior_image_0_left  → video.exterior_image_1_left
    observation/exterior_image_1_left  → video.exterior_image_2_left
    observation/wrist_image_left       → video.wrist_image_left
    observation/joint_position         → state.joint_position
    observation/gripper_position       → state.gripper_position
    prompt                             → annotation.language.language_instruction
"""

import argparse
import logging
import os
import pickle
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from einops import rearrange
import imageio

# Add eval_utils dir to path so policy_server can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add dreamzero root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from openpi_client.base_policy import BasePolicy
from tianshou.data import Batch

from policy_server import WebsocketPolicyServer, PolicyServerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Raise torch.compile recompile limit so variable-length inputs don't crash.
# local_benchmark.py uses 800; we match that here instead of disabling dynamo entirely.
import torch as _torch
_torch._dynamo.config.recompile_limit = 800


class DreamZeroPolicy(BasePolicy):
    """
    Wraps GrootSimPolicy to satisfy the WebsocketPolicyServer interface.

    Server calls:
      - policy.infer(obs)  → dict with 'actions': np.ndarray (H, 8)
      - policy.reset(reset_info)

    Observation keys received from DreamZeroJointPosClient (run_sim_eval.py):
      observation/exterior_image_0_left  (H, W, 3) uint8
      observation/exterior_image_1_left  (H, W, 3) uint8
      observation/wrist_image_left       (H, W, 3) uint8
      observation/joint_position         (7,) float64
      observation/gripper_position       (1,) float64
      prompt                             str
      session_id                         str
    """

    def __init__(self, groot_policy: GrootSimPolicy, video_out_dir: str = "/tmp/dreamzero_pred_videos"):
        self.policy = groot_policy
        self.video_out_dir = video_out_dir
        self._infer_count = 0
        os.makedirs(video_out_dir, exist_ok=True)
        logger.info(f"AI prediction videos will be saved to: {video_out_dir}")

        # Compare mode state
        self._last_video_pred = None  # saved for next chunk's async comparison

    def _build_policy_obs(self, obs: dict):
        """Build model input dict from client obs."""
        instruction = obs.get("prompt", "pick up the object")
        jp = obs["observation/joint_position"]
        gp = obs["observation/gripper_position"]
        if jp.ndim == 1:
            jp = jp[np.newaxis, :]
        if gp.ndim == 1:
            gp = gp[np.newaxis, :]
        return {
            "video.exterior_image_1_left":   obs["observation/exterior_image_0_left"],
            "video.exterior_image_2_left":   obs["observation/exterior_image_1_left"],
            "video.wrist_image_left":        obs["observation/wrist_image_left"],
            "state.joint_position":          jp,
            "state.gripper_position":        gp,
            "annotation.language.language_instruction":   instruction,
            "annotation.language.language_instruction_2": instruction,
            "annotation.language.language_instruction_3": instruction,
        }

    def _run_forward(self, policy_obs, latent_video=None):
        """Run model forward, broadcast to workers if TP active."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            device = f"cuda:{dist.get_rank()}"
            signal = torch.zeros(1, dtype=torch.int64, device=device)
            dist.broadcast(signal, src=0)
            serialized = pickle.dumps(policy_obs)
            size = torch.tensor([len(serialized)], dtype=torch.int64, device=device)
            dist.broadcast(size, src=0)
            data = torch.frombuffer(serialized, dtype=torch.uint8).to(device)
            dist.broadcast(data, src=0)

        with torch.no_grad():
            result_batch, video_pred = self.policy.lazy_joint_forward_causal(
                Batch(obs=policy_obs), latent_video=latent_video
            )
        return result_batch, video_pred

    def _extract_actions(self, result_batch):
        """Extract (H, 8) action array from model output."""
        act = result_batch.act
        action_dict = {}
        for k in dir(act):
            if k.startswith("action."):
                action_dict[k] = getattr(act, k)

        if not action_dict:
            return np.zeros((1, 8), dtype=np.float32)

        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().float().numpy()
            return np.array(x, dtype=np.float32)

        joint_pos = to_numpy(action_dict.get("action.joint_position", np.zeros((1, 7))))
        gripper_pos = to_numpy(action_dict.get("action.gripper_position", np.zeros((1, 1))))
        if joint_pos.ndim == 1:
            joint_pos = joint_pos.reshape(1, -1)
        if gripper_pos.ndim == 1:
            gripper_pos = gripper_pos.reshape(-1, 1)
        H = joint_pos.shape[0]
        if gripper_pos.shape[0] != H:
            gripper_pos = gripper_pos.reshape(H, -1)
        return np.concatenate([joint_pos, gripper_pos], axis=-1)

    def _get_kv_cache_seq_len(self):
        """Get current KV cache sequence length (from first layer)."""
        ah = self.policy.trained_model.action_head
        if ah.kv_cache1 and len(ah.kv_cache1) > 0:
            return ah.kv_cache1[0].shape[2]  # [2, B, seq_len, heads, dim]
        return 0

    def _truncate_kv_caches_local(self, target_len):
        """Truncate KV caches on this rank only."""
        ah = self.policy.trained_model.action_head
        for cache in [ah.kv_cache1, ah.kv_cache_neg, ah.crossattn_cache, ah.crossattn_cache_neg]:
            if cache is None:
                continue
            for i in range(len(cache)):
                cache[i] = cache[i][:, :, :target_len, :, :]
        ah.current_start_frame -= ah.num_frame_per_block

    def _truncate_kv_caches(self, target_len):
        """Truncate KV caches on ALL ranks (broadcast signal -2 to workers)."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            device = f"cuda:{dist.get_rank()}"
            # Signal -2 = truncate
            signal = torch.tensor([-2], dtype=torch.int64, device=device)
            dist.broadcast(signal, src=0)
            # Broadcast target length
            target = torch.tensor([target_len], dtype=torch.int64, device=device)
            dist.broadcast(target, src=0)
        # Truncate locally
        self._truncate_kv_caches_local(target_len)

    def _save_kv_state_all_ranks(self):
        """Save KV state on ALL ranks (broadcast signal -3 to workers).
        Returns saved state for rank 0; workers save to self._saved_kv_state."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            device = f"cuda:{dist.get_rank()}"
            signal = torch.tensor([-3], dtype=torch.int64, device=device)
            dist.broadcast(signal, src=0)
        return self.save_kv_state()

    def _restore_kv_state_all_ranks(self, state):
        """Restore KV state on ALL ranks (broadcast signal -4 to workers).
        Restores from provided state on rank 0; workers restore from self._saved_kv_state."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            device = f"cuda:{dist.get_rank()}"
            signal = torch.tensor([-4], dtype=torch.int64, device=device)
            dist.broadcast(signal, src=0)
        self.restore_kv_state(state)

    def infer(self, obs: dict) -> dict:
        t_infer_start = time.perf_counter()

        # Allow client to reset server state at episode boundaries
        if obs.get("reset_episode", False):
            self._last_video_pred = None
            logger.info("[reset_episode] Cleared _last_video_pred for new episode")

        # Allow client to switch output directory per run without restarting server
        if "output_dir" in obs and obs["output_dir"] != self.video_out_dir:
            self.video_out_dir = obs["output_dir"]
            self._infer_count = 0
            os.makedirs(self.video_out_dir, exist_ok=True)
            logger.info(f"[output_dir] switched to: {self.video_out_dir}")

        policy_obs = self._build_policy_obs(obs)
        compare_mode = obs.get("compare_mode", False)
        predicted_mode = obs.get("predicted_mode", False)

        t_prep = time.perf_counter()

        action_async = None

        if compare_mode and self._last_video_pred is not None:
            # === Compare mode: run async + sync, return both ===
            saved_state = self._save_kv_state_all_ranks()

            result_async, _ = self._run_forward(policy_obs, latent_video=self._last_video_pred)
            action_async = self._extract_actions(result_async)
            logger.info(f"[compare] async inference done (used video_pred as latent_video)")

            self._restore_kv_state_all_ranks(saved_state)

        t_async = time.perf_counter()

        if predicted_mode and self._last_video_pred is not None:
            # === Predicted mode: use latent_video at non-reset chunks ===
            # At csf=0 (reset), pass None so real observation is used (static frame).
            ah = self.policy.trained_model.action_head
            at_reset = (ah.current_start_frame == 0 or
                        ah.current_start_frame >= ah.model.local_attn_size)
            lv = None if at_reset else self._last_video_pred
            if lv is not None:
                logger.info(f"[predicted] Using latent_video (csf={ah.current_start_frame})")
            else:
                logger.info(f"[predicted] Reset chunk (csf={ah.current_start_frame}), using real obs")
            result_batch, video_pred = self._run_forward(policy_obs, latent_video=lv)
        else:
            # === Sync mode (default): use real obs ===
            result_batch, video_pred = self._run_forward(policy_obs)

        actions = self._extract_actions(result_batch)

        t_forward = time.perf_counter()

        # Save video_pred for next chunk
        self._last_video_pred = video_pred

        # Decode and save AI prediction video
        self._save_pred_video(video_pred)
        t_pred_video = time.perf_counter()

        # Save full action array alongside prediction video
        self._save_actions(actions)

        t_end = time.perf_counter()
        async_time = t_async - t_prep if action_async is not None else 0
        logger.info(
            f"[infer timing] prep={t_prep - t_infer_start:.3f}s, "
            f"async={async_time:.3f}s, "
            f"forward={t_forward - t_async:.3f}s, "
            f"pred_video={t_pred_video - t_forward:.3f}s, "
            f"save={t_end - t_pred_video:.3f}s, "
            f"TOTAL={t_end - t_infer_start:.3f}s"
            f"{' [predicted]' if predicted_mode else ''}"
        )

        result = {"actions": actions}
        if action_async is not None:
            result["actions_async"] = action_async
        return result

    def _save_pred_video(self, video_pred) -> None:
        """Decode VAE latent and save AI prediction video to disk."""
        if video_pred is None:
            return
        try:
            action_head = self.policy.trained_model.action_head
            with torch.no_grad():
                frames = action_head.vae.decode(
                    video_pred,
                    tiled=action_head.tiled,
                    tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                    tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
                )
            # (B, C, T, H, W) → (T, H, W, C), pixel range [0, 255]
            frames = rearrange(frames, "B C T H W -> B T H W C")
            frames = ((frames[0].float() + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)

            out_path = os.path.join(self.video_out_dir, f"pred_{self._infer_count:05d}.mp4")
            imageio.mimsave(out_path, list(frames), fps=15, codec="libx264")
            logger.info(f"[pred video] saved {len(frames)} frames → {out_path}")
        except Exception as e:
            logger.warning(f"[pred video] failed to save: {e}")
        finally:
            self._infer_count += 1

    def _save_actions(self, actions: np.ndarray) -> None:
        """Save full action array (H, 8) as .npy and human-readable .txt."""
        idx = self._infer_count - 1  # _infer_count already incremented in _save_pred_video
        try:
            npy_path = os.path.join(self.video_out_dir, f"pred_{idx:05d}_actions.npy")
            txt_path = os.path.join(self.video_out_dir, f"pred_{idx:05d}_actions.txt")

            np.save(npy_path, actions)

            # Human-readable: header + rows
            H = actions.shape[0]
            with open(txt_path, "w") as f:
                f.write(f"# chunk {idx:05d}  shape=({H}, 8)\n")
                f.write(f"# cols: j0  j1  j2  j3  j4  j5  j6  gripper\n")
                for step, row in enumerate(actions):
                    vals = "  ".join(f"{v:+.4f}" for v in row)
                    f.write(f"step {step:02d}: {vals}\n")

            # Also log gripper column to server log for quick inspection
            gripper_vals = actions[:, -1]
            logger.info(
                f"[actions {idx:05d}] gripper={np.array2string(gripper_vals, precision=3, separator=',')}  "
                f"joint_mean={actions[:, :7].mean(axis=0)}"
            )
        except Exception as e:
            logger.warning(f"[actions] failed to save: {e}")

    def _clone_kv_cache_to_cpu(self, cache):
        """Clone a KV cache to CPU memory to avoid GPU OOM."""
        if cache is None:
            return None
        if isinstance(cache, list):
            return [self._clone_kv_cache_to_cpu(item) for item in cache]
        if isinstance(cache, torch.Tensor):
            return cache.detach().cpu().clone()
        return cache

    def _move_kv_cache_to_gpu(self, cache, device):
        """Move a CPU KV cache back to GPU."""
        if cache is None:
            return None
        if isinstance(cache, list):
            return [self._move_kv_cache_to_gpu(item, device) for item in cache]
        if isinstance(cache, torch.Tensor):
            return cache.to(device)
        return cache

    def save_kv_state(self):
        """Save model's KV cache state to CPU memory."""
        ah = self.policy.trained_model.action_head
        return {
            "kv_cache1": self._clone_kv_cache_to_cpu(ah.kv_cache1),
            "kv_cache_neg": self._clone_kv_cache_to_cpu(ah.kv_cache_neg),
            "crossattn_cache": self._clone_kv_cache_to_cpu(ah.crossattn_cache),
            "crossattn_cache_neg": self._clone_kv_cache_to_cpu(ah.crossattn_cache_neg),
            "current_start_frame": ah.current_start_frame,
            "clip_feas": ah.clip_feas.detach().cpu().clone() if ah.clip_feas is not None else None,
            "ys": ah.ys.detach().cpu().clone() if ah.ys is not None else None,
            "language": ah.language.detach().cpu().clone() if ah.language is not None else None,
        }

    def restore_kv_state(self, state):
        """Restore model's KV cache state from CPU back to GPU."""
        ah = self.policy.trained_model.action_head
        device = ah._device
        ah.kv_cache1 = self._move_kv_cache_to_gpu(state["kv_cache1"], device)
        ah.kv_cache_neg = self._move_kv_cache_to_gpu(state["kv_cache_neg"], device)
        ah.crossattn_cache = self._move_kv_cache_to_gpu(state["crossattn_cache"], device)
        ah.crossattn_cache_neg = self._move_kv_cache_to_gpu(state["crossattn_cache_neg"], device)
        ah.current_start_frame = state["current_start_frame"]
        ah.clip_feas = state["clip_feas"].to(device) if state["clip_feas"] is not None else None
        ah.ys = state["ys"].to(device) if state["ys"] is not None else None
        ah.language = state["language"].to(device) if state["language"] is not None else None

    def infer_with_latent_video(self, obs: dict, latent_video=None) -> dict:
        """Run inference with optional latent_video (for async simulation)."""
        instruction = obs.get("prompt", "pick up the object")
        jp = obs["observation/joint_position"]
        gp = obs["observation/gripper_position"]
        if jp.ndim == 1:
            jp = jp[np.newaxis, :]
        if gp.ndim == 1:
            gp = gp[np.newaxis, :]

        policy_obs = {
            "video.exterior_image_1_left": obs["observation/exterior_image_0_left"],
            "video.exterior_image_2_left": obs["observation/exterior_image_1_left"],
            "video.wrist_image_left": obs["observation/wrist_image_left"],
            "state.joint_position": jp,
            "state.gripper_position": gp,
            "annotation.language.language_instruction": instruction,
            "annotation.language.language_instruction_2": instruction,
            "annotation.language.language_instruction_3": instruction,
        }

        with torch.no_grad():
            result_batch, video_pred = self.policy.lazy_joint_forward_causal(
                Batch(obs=policy_obs), latent_video=latent_video
            )

        act = result_batch.act
        action_dict = {}
        for k in dir(act):
            if k.startswith("action."):
                action_dict[k] = getattr(act, k)

        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().float().numpy()
            return np.array(x, dtype=np.float32)

        joint_pos = to_numpy(action_dict.get("action.joint_position", np.zeros((1, 7))))
        gripper_pos = to_numpy(action_dict.get("action.gripper_position", np.zeros((1, 1))))
        if joint_pos.ndim == 1:
            joint_pos = joint_pos.reshape(1, -1)
        if gripper_pos.ndim == 1:
            gripper_pos = gripper_pos.reshape(-1, 1)
        H = joint_pos.shape[0]
        if gripper_pos.shape[0] != H:
            gripper_pos = gripper_pos.reshape(H, -1)
        actions = np.concatenate([joint_pos, gripper_pos], axis=-1)

        return {"actions": actions, "video_pred": video_pred}

    def reset(self, reset_info: dict) -> None:
        self._last_video_pred = None
        logger.info("Episode reset.")


def main():
    parser = argparse.ArgumentParser(description="DreamZero policy WebSocket server for Isaac Lab sim eval")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/fact_data/qiliu/dreamzero_weights/DreamZero-DROID",
        help="Path to DreamZero model checkpoint directory",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--pred-video-dir",
        type=str,
        default="/fact_data/qiliu/worldmodel_data/sim-evals/runs/pred_videos",
        help="Directory to save AI prediction videos (decoded from VAE latents)",
    )
    args = parser.parse_args()

    # ── Distributed / TP setup ─────────────────────────────────────────────────
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        if world_size <= 2:
            device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("ip",))
        else:
            tp_size = world_size // 2
            device_mesh = init_device_mesh("cuda", mesh_shape=(2, tp_size), mesh_dim_names=("ip", "tp"))
        logger.info(f"Rank {rank}/{world_size}, mesh ready: {device_mesh}")
        # Eagerly initialize all sub-mesh NCCL communicators BEFORE model loading.
        # If deferred to model.parallelize() (after weights are on GPU), the NCCL
        # comm init can hang on some H20 driver/firmware combinations.
        # We must also perform a dummy all_reduce to force the underlying NCCL
        # communicator to be fully created (get_group() alone only creates the
        # Python ProcessGroup; the CUDA-level comm is lazy-initialized on first op).
        # Additionally, we pre-warm functional collectives (used by torch.compile
        # with fullgraph=True) so they reuse existing comms instead of creating new ones.
        import torch.distributed._functional_collectives as funcol
        logger.info(f"[Rank {rank}] Pre-initializing mesh sub-communicators...")
        ip_group = device_mesh["ip"].get_group()
        dummy = torch.zeros(1, device=f"cuda:{rank}")
        dist.all_reduce(dummy, group=ip_group)
        funcol.all_reduce(dummy, "sum", ip_group)
        logger.info(f"[Rank {rank}] ip sub-mesh ready (rank={device_mesh['ip'].get_local_rank()}, size={device_mesh['ip'].size()})")
        if "tp" in device_mesh.mesh_dim_names:
            tp_group = device_mesh["tp"].get_group()
            dist.all_reduce(dummy, group=tp_group)
            funcol.all_reduce(dummy, "sum", tp_group)
            logger.info(f"[Rank {rank}] tp sub-mesh ready (rank={device_mesh['tp'].get_local_rank()}, size={device_mesh['tp'].size()})")
        del dummy
    else:
        # GrootSimPolicy unconditionally calls dist.get_rank(), so we must
        # initialize process group even for single-GPU inference.
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        dist.init_process_group("nccl", world_size=1, rank=0)
        rank = 0
        device_mesh = None

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    # ── Load model ─────────────────────────────────────────────────────────────
    logger.info(f"[Rank {rank}] Loading DreamZero-DROID from {args.model_path} ...")
    groot_policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("oxe_droid"),
        model_path=args.model_path,
        device=device,
        device_mesh=device_mesh,
    )
    logger.info(f"[Rank {rank}] Model loaded.")

    policy = DreamZeroPolicy(groot_policy, video_out_dir=args.pred_video_dir)

    # ── Server (rank 0 only) ───────────────────────────────────────────────────
    # PolicyServerConfig tells the client what the server expects.
    # The DreamZeroJointPosClient in run_sim_eval.py sends:
    #   2 external cameras + 1 wrist camera + joint_pos + gripper_pos
    server_cfg = PolicyServerConfig(
        image_resolution=None,       # client resizes before sending
        needs_wrist_camera=True,
        n_external_cameras=2,        # exterior_image_0_left + exterior_image_1_left
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="joint_position",
    )

    if rank == 0:
        logger.info(f"Starting WebSocket server at {args.host}:{args.port}")
        server = WebsocketPolicyServer(policy, server_cfg, host=args.host, port=args.port)
        server.serve_forever()
    elif world_size > 1:
        # Non-rank-0 processes must participate in every forward pass for TP.
        # Signals: 0 = run inference, -1 = shutdown, -2 = truncate KV cache,
        #          -3 = save KV state, -4 = restore KV state
        logger.info(f"Rank {rank} entering TP worker loop...")
        _saved_kv_state = None
        while True:
            signal = torch.zeros(1, dtype=torch.int64, device=device)
            dist.broadcast(signal, src=0)
            if signal.item() == -1:
                logger.info(f"Rank {rank} received shutdown signal.")
                break
            elif signal.item() == -2:
                # Truncate KV cache
                target = torch.zeros(1, dtype=torch.int64, device=device)
                dist.broadcast(target, src=0)
                policy._truncate_kv_caches_local(int(target.item()))
                continue
            elif signal.item() == -3:
                # Save KV state locally
                _saved_kv_state = policy.save_kv_state()
                continue
            elif signal.item() == -4:
                # Restore KV state locally
                if _saved_kv_state is not None:
                    policy.restore_kv_state(_saved_kv_state)
                    _saved_kv_state = None
                continue
            # Signal 0: run inference
            size = torch.zeros(1, dtype=torch.int64, device=device)
            dist.broadcast(size, src=0)
            data = torch.zeros(size.item(), dtype=torch.uint8, device=device)
            dist.broadcast(data, src=0)
            obs = pickle.loads(data.cpu().numpy().tobytes())
            batch = Batch(obs=obs)
            with torch.no_grad():
                policy.policy.lazy_joint_forward_causal(batch)
    else:
        logger.error("Single-GPU but rank != 0, this should not happen.")
        sys.exit(1)


if __name__ == "__main__":
    main()
