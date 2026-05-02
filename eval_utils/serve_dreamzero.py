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

        # Frame buffering — mirrors clean ARDroidRoboarenaPolicy behavior so the
        # action_head receives videos.shape[2] = FRAMES_PER_CHUNK (default 4)
        # instead of a single frame each chunk. Provides real temporal diversity
        # to recondition VAE input + KV cache content.
        # FRAMES_PER_CHUNK=1 disables buffering (legacy single-frame behavior).
        self.FRAMES_PER_CHUNK = int(os.environ.get("FRAMES_PER_CHUNK", "4"))
        # ASYNC_LAG_CHUNKS=N simulates async inference: model sees obs from N
        # chunks ago (i.e. one chunk's actions are conceptually still executing
        # while this prediction runs). Lags BOTH video buffer and joint/gripper
        # state. Default 0 = sync.
        self.ASYNC_LAG_CHUNKS = int(os.environ.get("ASYNC_LAG_CHUNKS", "0"))
        # ASYNC_PRED_AS_OBS=1: at each inference, replace the LATEST entry of
        # the model's input video buffer with the last frame of the previously
        # predicted video (split per camera from the 2x2 composite). Older
        # entries remain real obs. Effectively: model uses its own prior
        # prediction as a stand-in for "obs at the moment we are predicting"
        # (matches true-async semantics where the prior chunk's actions are
        # still executing). Each subsequent chunk naturally "corrects" the
        # prior pred-stand-in by shifting it back into the real-obs region.
        self.ASYNC_PRED_AS_OBS = int(os.environ.get("ASYNC_PRED_AS_OBS", "0"))
        self._last_pred_composite = None  # last decoded pred frame (H, W, 3) uint8
        # ASYNC_QUEUE=1 + ASYNC_PRED_AS_OBS=1: properly time-shift B's actions.
        # B predicts the chunk AFTER the current one; queue it so it gets
        # returned at the next infer call (= applied at chunk K+1 window).
        # Chunk 0 falls back to sync (no pending yet).
        self.ASYNC_QUEUE = int(os.environ.get("ASYNC_QUEUE", "0"))
        self._pending_actions = None     # queued actions to return next call
        self._pending_video_pred = None  # queued video latent (saved as pred_NNNNN.mp4 next call)
        # ASYNC_FALLBACK_SQ_THRESH: if sum_j (A_K[0, j] - prev_B[-1, j])^2 > threshold,
        # fall back to A_K (not B_shifted) for this chunk. Detects severe A↔prev-B
        # disagreement which signals B's OOD bias is too large for shift to fix.
        # Default 1e9 = effectively disabled.
        self.ASYNC_FALLBACK_SQ_THRESH = float(os.environ.get("ASYNC_FALLBACK_SQ_THRESH", "1e9"))
        # B_AMPLITUDE_SCALE=1: stretch B's per-chunk per-joint amplitude to match
        # A's slope at the chunk boundary. scale_j = vA / vB, where with window W,
        # vA = A[-1,j]-A[-1-W,j] and vB = B[W,j]-B[0,j]. W=1 = single-step (noisy);
        # W=3 = 3-step finite diff (smoother, suppresses scale outliers).
        # Applied as B_new[:, j] = B[0, j] + (B[:, j] - B[0, j]) * scale_j, then clip
        # the resulting delta to ±A_range_j (so B can't be wilder than A).
        self.B_AMPLITUDE_SCALE = int(os.environ.get("B_AMPLITUDE_SCALE", "0"))
        self.B_AMPLITUDE_SCALE_WINDOW = int(os.environ.get("B_AMPLITUDE_SCALE_WINDOW", "1"))
        # FORCE_PAUSE_EVERY_N: bypass ASYNC_FALLBACK_SQ_THRESH; force a pause
        # every N two-step entries since the last pause. After a pause, the next
        # chunk runs SYNC (chunk-0-like) and the counter resets. 0 = disabled.
        self.FORCE_PAUSE_EVERY_N = int(os.environ.get("FORCE_PAUSE_EVERY_N", "0"))
        self._two_step_count_since_pause = 0
        # GRIP_TRANSITION_PAUSE: when the current chunk's gripper command goes
        # from open (<=0.2) to closed (>0.2), force a pause on the NEXT chunk.
        # Reason: the open→closed transition = grasping moment. Right after,
        # (a) wrist camera view changes drastically (object now in gripper),
        # (b) sim PID needs settling time for the gripper to physically close,
        # (c) PD1's prior bias was learned on "not holding object" dynamics.
        # The pause + chunk-0-like restart gives the next plan fresh real-obs
        # of "holding object" state to start from.
        self.GRIP_TRANSITION_PAUSE = int(os.environ.get("GRIP_TRANSITION_PAUSE", "1"))
        self._pending_grip_pause = False
        # GRIP_PRE_PAUSE: A (lagged real-obs) predicted gripper close anywhere
        # in chunk K-1 but actual prev_applied gripper ended open → A reports
        # a missed grasp. Pause this chunk to fix and prevent error buildup.
        self.GRIP_PRE_PAUSE = int(os.environ.get("GRIP_PRE_PAUSE", "1"))
        self._prev_applied_last_gripper = None
        # GRIP_B_ABORT: in two-step path, if current chunk's B raw plan is
        # open→close (a grasp attempt), abort this chunk (hold-still) and let
        # the next post-pause chunk-0-like sync handle the grasp. Principle:
        # only post-pause sync chunks should perform grasp actions, never
        # async two-step chunks (whose B is corrupted by VAE round-trip).
        self.GRIP_B_ABORT = int(os.environ.get("GRIP_B_ABORT", "1"))
        # DISTILL_MODEL_PATH: path to a trained ResidualConv1D checkpoint that
        # corrects B_raw using a learned VAE-bias residual. If set, applied in
        # two-step path before shift+scale+PD1. Replaces PD1 mechanism.
        self._distill_model = None
        distill_path = os.environ.get("DISTILL_MODEL_PATH", "").strip()
        if distill_path and os.path.exists(distill_path):
            try:
                import sys
                sys.path.insert(0, "/fact_home/qiliu/worldmodel/distill")
                from model import ResidualConv1D
                ckpt = torch.load(distill_path, map_location="cpu", weights_only=False)
                margs = ckpt.get("args", {})
                self._distill_model = ResidualConv1D(
                    in_ch=24, hid=margs.get("hid", 64),
                    out_ch=7, n_layers=margs.get("n_layers", 3))
                self._distill_model.load_state_dict(ckpt["state_dict"])
                self._distill_model.eval()
                logger.info(f"[distill] loaded model from {distill_path} "
                            f"(val_mse={ckpt.get('val_mse', '?')}, ep={ckpt.get('epoch', '?')})")
            except Exception as e:
                logger.warning(f"[distill] failed to load model: {e}")
                self._distill_model = None
        # B_PREV_DIFF_ALPHA: residual correction using (A_K - B_{K-1}) carried
        # from the previous chunk. Both A_K and B_{K-1} predict chunk K-1; the
        # diff is the VAE round-trip bias on chunk K-1. Apply alpha * diff to
        # B_K (current chunk) on dims 0..6 (gripper untouched).
        self.B_PREV_DIFF_ALPHA = float(os.environ.get("B_PREV_DIFF_ALPHA", "0.0"))
        self._prev_applied_full = None   # last applied chunk's full action array (H,8)
        # RELATIVE_ACTION_SCALE: amplify per-chunk relative joint actions before
        # returning. action_t' = current_jp + s * (action_t - current_jp), applied
        # to j0..j6 only. gripper untouched. 1.0 = identity. Tests hypothesis that
        # model under-shoots actual motion magnitude.
        self.RELATIVE_ACTION_SCALE = float(os.environ.get("RELATIVE_ACTION_SCALE", "1.0"))
        self._prev_applied_last = None   # last step of last applied chunk's actions (j0..j6)
        # When fallback triggers, we emit hold-still actions and set this flag so the
        # NEXT chunk acts like chunk 0: clear frame/state buffers, single-frame mode,
        # no lag. KV cache is anyway reset every chunk under FORCE_RESET_EVERY_CHUNK=1.
        self._pending_reset = False
        self._frame_buffers: dict[str, list] = {
            "video.exterior_image_1_left":  [],
            "video.exterior_image_2_left":  [],
            "video.wrist_image_left":       [],
        }
        self._state_buffers: dict[str, list] = {"jp": [], "gp": []}
        self._is_first_call = True
        logger.info(f"Frame buffering: FRAMES_PER_CHUNK={self.FRAMES_PER_CHUNK} ASYNC_LAG_CHUNKS={self.ASYNC_LAG_CHUNKS} ASYNC_PRED_AS_OBS={self.ASYNC_PRED_AS_OBS} ASYNC_FALLBACK_SQ_THRESH={self.ASYNC_FALLBACK_SQ_THRESH}")

    def _split_pred_composite(self, composite, target_h, target_w):
        """Split a (H_comp, W_comp, 3) pred composite back into 3 camera views.

        Composite layout (from dreamzero_cotrain._prepare_video DROID branch):
          - Top half [0:H, :]:   wrist, stretched 2x wider via np.repeat axis=-1
          - Bottom-left [H:, :W]: left exterior  (eval's exterior_image_0_left)
          - Bottom-right [H:, W:]: right exterior (eval's exterior_image_1_left)

        Returns dict with keys matching eval's observation/* names, each
        resized to (target_h, target_w, 3).
        """
        import numpy as np
        from PIL import Image

        H_comp, W_comp = composite.shape[:2]
        h, w = H_comp // 2, W_comp // 2

        # Wrist: top half, sample every other column to undo 2x stretch
        wrist = composite[:h, ::2, :]   # (h, w, 3)
        # Bottom-left = left exterior (= eval's exterior_image_0_left per serve_dreamzero header)
        left_ext = composite[h:, :w, :]
        # Bottom-right = right exterior (= eval's exterior_image_1_left)
        right_ext = composite[h:, w:, :]

        def _resize(img, h_out, w_out):
            if img.shape[:2] == (h_out, w_out):
                return img
            return np.array(Image.fromarray(img).resize((w_out, h_out), Image.BILINEAR))

        return {
            "observation/exterior_image_0_left": _resize(left_ext, target_h, target_w),
            "observation/exterior_image_1_left": _resize(right_ext, target_h, target_w),
            "observation/wrist_image_left":      _resize(wrist, target_h, target_w),
        }

    def _decode_pred_last_frame(self, video_pred):
        """Decode VAE latent to pixel composite, return last frame as uint8."""
        action_head = self.policy.trained_model.action_head
        with torch.no_grad():
            frames = action_head.vae.decode(
                video_pred,
                tiled=action_head.tiled,
                tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
            )
        frames = rearrange(frames, "B C T H W -> B T H W C")
        frames = ((frames[0].float() + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
        return frames[-1].copy()

    def _add_label(self, frame, label):
        """Overlay a colored corner rectangle with text label on a frame."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            return frame
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        fill_color = {
            "A": (220, 60, 60),    # red = sync inference (step A)
            "B": (60, 200, 60),    # green = final 2-step output (step B)
            "T": (60, 100, 220),   # blue = truth (ground-truth episode slice)
            "I": (200, 160, 50),   # amber = input buffer to forward A
            "J": (180, 120, 200),  # purple = input buffer to forward B
        }.get(label, (100, 100, 200))
        rect_w, rect_h = 80, 50
        draw.rectangle([0, 0, rect_w, rect_h], fill=fill_color)
        font = None
        for fp in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            try:
                font = ImageFont.truetype(fp, 36)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
        draw.text((22, 4), label, fill=(255, 255, 255), font=font)
        return np.array(img)

    def _save_truth_video(self, policy_obs, filename, label="T"):
        """Save the 3-camera buffer in policy_obs as a 2x2 composite mp4 that
        mirrors the model's _prepare_video DROID layout (wrist top stretched,
        left+right exterior bottom)."""
        try:
            left_ext = policy_obs.get("video.exterior_image_1_left")
            right_ext = policy_obs.get("video.exterior_image_2_left")
            wrist = policy_obs.get("video.wrist_image_left")
            if left_ext is None or left_ext.ndim != 4:
                return
            T, H, W, _ = left_ext.shape
            comp = np.zeros((T, 2 * H, 2 * W, 3), dtype=np.uint8)
            comp[:, :H, :, :] = np.repeat(wrist, 2, axis=2)   # wrist stretched 2x wide
            comp[:, H:, :W, :] = left_ext                      # bottom-left
            comp[:, H:, W:, :] = right_ext                     # bottom-right
            if label is not None:
                comp = np.stack([self._add_label(f, label) for f in comp], axis=0)
            out_path = os.path.join(self.video_out_dir, filename)
            imageio.mimsave(out_path, list(comp), fps=15, codec="libx264")
            logger.info(f"[truth video] saved {len(comp)} frames (label={label}) → {out_path}")
        except Exception as e:
            logger.warning(f"[truth video] failed: {e}")

    def _save_pred_video_named(self, video_pred, filename, label=None):
        """Decode + save a pred video under a custom filename. Optionally overlay
        a colored A/B label in the top-left corner of every frame. Returns the
        UNLABELED last frame (so callers can use it for split/swap)."""
        if video_pred is None:
            return None
        try:
            action_head = self.policy.trained_model.action_head
            with torch.no_grad():
                frames = action_head.vae.decode(
                    video_pred,
                    tiled=action_head.tiled,
                    tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                    tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
                )
            frames = rearrange(frames, "B C T H W -> B T H W C")
            frames = ((frames[0].float() + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
            last_frame_unlabeled = frames[-1].copy()
            if label is not None:
                frames = np.stack([self._add_label(f, label) for f in frames], axis=0)
            out_path = os.path.join(self.video_out_dir, filename)
            imageio.mimsave(out_path, list(frames), fps=15, codec="libx264")
            logger.info(f"[pred video] saved {len(frames)} frames (label={label}) → {out_path}")
            return last_frame_unlabeled
        except Exception as e:
            logger.warning(f"[pred video named={filename}] failed: {e}")
            return None

    def _swap_latest_to_pred(self, policy_obs, pred_composite):
        """Legacy single-frame swap. Superseded by `_build_b_obs_from_pred` for
        the two-step pipeline (which now feeds B the full 9-frame decoded
        prediction). Kept for backward compatibility / experimentation."""
        ref = policy_obs["video.exterior_image_1_left"]
        if ref.ndim != 4 or ref.shape[0] < 1:
            return policy_obs
        tgt_h, tgt_w = ref.shape[1], ref.shape[2]
        split = self._split_pred_composite(pred_composite, tgt_h, tgt_w)
        mapping = {
            "video.exterior_image_1_left":  split["observation/exterior_image_0_left"],
            "video.exterior_image_2_left":  split["observation/exterior_image_1_left"],
            "video.wrist_image_left":       split["observation/wrist_image_left"],
        }
        new_obs = dict(policy_obs)
        for k, frame in mapping.items():
            stacked = new_obs[k]
            new_obs[k] = np.concatenate([stacked[1:], frame[None]], axis=0)
        # Optional debug dump
        if os.environ.get("DUMP_SPLIT_PRED", "0") == "1":
            try:
                debug_dir = os.path.join(self.video_out_dir, "split_debug")
                os.makedirs(debug_dir, exist_ok=True)
                idx = self._infer_count
                imageio.imwrite(os.path.join(debug_dir, f"chunk{idx:05d}_composite.png"), pred_composite)
                for k, img in mapping.items():
                    imageio.imwrite(os.path.join(debug_dir, f"chunk{idx:05d}_{k.replace('.', '_')}.png"), img)
            except Exception as e:
                logger.warning(f"[DUMP_SPLIT_PRED] failed: {e}")
        return new_obs

    def _decode_pred_to_frames(self, video_pred):
        """Decode a VAE latent to a (T, H_comp, W_comp, 3) uint8 ndarray.
        T is the model's native temporal output (typically 9 raw frames per
        chunk). Returns None on failure."""
        if video_pred is None:
            return None
        try:
            action_head = self.policy.trained_model.action_head
            with torch.no_grad():
                frames = action_head.vae.decode(
                    video_pred,
                    tiled=action_head.tiled,
                    tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                    tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
                )
            frames = rearrange(frames, "B C T H W -> B T H W C")
            return ((frames[0].float() + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
        except Exception as e:
            logger.warning(f"[decode pred] failed: {e}")
            return None

    def _build_b_obs_from_pred(self, policy_obs_a, pred_a):
        """Construct B's policy_obs by replacing the entire video time axis with
        A's full decoded prediction (typically 9 frames per camera). The state
        and language fields are inherited from A's policy_obs.

        Rationale: A's prediction represents one whole chunk of execution
        (9 raw frames / 3 motion latents). Reusing only `pred_A_last` (the
        legacy `_swap_latest_to_pred` behaviour) compresses 1 chunk's worth of
        information into 1 buffer slot. Feeding the full 9 frames lets the
        action_head go through its native shape[2]==9 path (skipping the
        repeat_interleave that shape[2]==4 would trigger), preserving all of
        A's predicted motion as B's "observed" chunk.

        Accepts both ndim=3 (single-frame, e.g. chunk 1 boundary where A
        degraded to 1-frame) and ndim=4 (multi-frame) policy_obs_a. In both
        cases B receives the full decoded prediction (9 frames)."""
        ref = policy_obs_a.get("video.exterior_image_1_left")
        if ref is None:
            return policy_obs_a
        if ref.ndim == 4:
            tgt_h, tgt_w = ref.shape[1], ref.shape[2]
        elif ref.ndim == 3:
            tgt_h, tgt_w = ref.shape[0], ref.shape[1]
        else:
            return policy_obs_a

        composite_frames = self._decode_pred_to_frames(pred_a)
        if composite_frames is None or composite_frames.shape[0] < 1:
            logger.warning("[B obs] decode failed, falling back to A obs unchanged")
            return policy_obs_a

        # Split each composite frame per-camera; stack along time axis.
        per_cam = {
            "video.exterior_image_1_left":  [],
            "video.exterior_image_2_left":  [],
            "video.wrist_image_left":       [],
        }
        for f in composite_frames:
            split = self._split_pred_composite(f, tgt_h, tgt_w)
            per_cam["video.exterior_image_1_left"].append(split["observation/exterior_image_0_left"])
            per_cam["video.exterior_image_2_left"].append(split["observation/exterior_image_1_left"])
            per_cam["video.wrist_image_left"].append(split["observation/wrist_image_left"])

        new_obs = dict(policy_obs_a)
        for k, frames_list in per_cam.items():
            new_obs[k] = np.stack(frames_list, axis=0)  # (T=9, H, W, 3)

        if os.environ.get("DUMP_SPLIT_PRED", "0") == "1":
            try:
                debug_dir = os.path.join(self.video_out_dir, "split_debug")
                os.makedirs(debug_dir, exist_ok=True)
                idx = self._infer_count
                for t, f in enumerate(composite_frames):
                    imageio.imwrite(
                        os.path.join(debug_dir, f"chunk{idx:05d}_compositeT{t:02d}.png"), f
                    )
            except Exception as e:
                logger.warning(f"[DUMP_SPLIT_PRED] failed: {e}")

        return new_obs

    def _build_policy_obs(self, obs: dict):
        """Build model input dict from client obs.

        Uses frame buffering: appends current frame to each per-camera buffer,
        then sends the most recent FRAMES_PER_CHUNK frames as (T, H, W, 3) arrays.
        First call sends T=1, subsequent calls send T=FRAMES_PER_CHUNK.
        """
        instruction = obs.get("prompt", "pick up the object")
        jp = obs["observation/joint_position"]
        gp = obs["observation/gripper_position"]
        if jp.ndim == 1:
            jp = jp[np.newaxis, :]
        if gp.ndim == 1:
            gp = gp[np.newaxis, :]

        # Reset buffers on episode boundary (client sends reset_episode=True at chunk 0)
        # OR on server-side pending reset (fallback triggered last chunk → behave
        # exactly like chunk 0: empty buffers, single-frame, lag bypassed for this call).
        is_fresh_start = False
        if obs.get("reset_episode", False) or self._pending_reset:
            for k in self._frame_buffers:
                self._frame_buffers[k].clear()
            for k in self._state_buffers:
                self._state_buffers[k].clear()
            self._is_first_call = True
            self._last_pred_composite = None
            if self._pending_reset:
                is_fresh_start = True
                self._pending_reset = False
                logger.info("[pending_reset] consumed: this chunk treated as fresh chunk-0 (no lag, single-frame)")

        # Append latest frame from each camera. Mirrors clean's
        # socket_test_optimized_AR.py:120-125: ndim=4 obs (multi-frame batch
        # from a single chunk window) gets `extend`ed; ndim=3 single frame
        # gets `append`ed.
        cam_inputs = {
            "video.exterior_image_1_left":  obs["observation/exterior_image_0_left"],
            "video.exterior_image_2_left":  obs["observation/exterior_image_1_left"],
            "video.wrist_image_left":       obs["observation/wrist_image_left"],
        }
        for k, data in cam_inputs.items():
            if isinstance(data, np.ndarray) and data.ndim == 4:
                if is_fresh_start:
                    # Post-pause restart: behave exactly like chunk 0 — push
                    # only the most recent frame (offset 0) so buf_len=1 and
                    # `do_async_two_step` (which requires buf_len>1) skips the
                    # A+B two-step path. Avoids B's shift/scale dance on a
                    # state with no real history; single forward pass on the
                    # current frame gives a clean K=0-like action chunk.
                    self._frame_buffers[k].append(data[-1])
                else:
                    self._frame_buffers[k].extend(list(data))
            else:
                self._frame_buffers[k].append(data)
        self._state_buffers["jp"].append(jp)
        self._state_buffers["gp"].append(gp)

        # Determine how many frames to send to the model this chunk
        if self._is_first_call or self.FRAMES_PER_CHUNK == 1:
            num_frames = 1
            self._is_first_call = False
        else:
            num_frames = self.FRAMES_PER_CHUNK

        buf_len = len(self._frame_buffers["video.exterior_image_1_left"])

        # ASYNC_LAG_CHUNKS=N means "skip the most recent N chunks' worth of
        # frames" (= N * num_frames buffer entries since each non-first chunk
        # extends num_frames frames per push). This matches true-async
        # semantics: inference K starts at T_{K-1} and cannot see frames
        # captured DURING chunk K-1's execution (the most recent push to the
        # buffer). Chunk K=lag is the boundary: the lagged window points
        # before any 4-frame push existed (only chunk 0's single reset frame
        # is in range). Padding by repeating buf[0] would create a 4-frame
        # static buffer that's out-of-distribution for the model. Instead,
        # degrade to 1-frame mode using buf[0] (chunk 0's reset frame) — this
        # mirrors sync chunk 0's input exactly, which is the natural
        # interpretation of "K chunks ago = chunk 0's reset state".
        lag = self.ASYNC_LAG_CHUNKS
        lag_entries = lag * num_frames
        if not is_fresh_start and lag > 0 and buf_len > lag_entries:
            if lag_entries + num_frames > buf_len:
                # Underflow at the chunk-K=lag boundary: not enough history
                # for `num_frames` distinct lagged frames. Degrade to 1-frame
                # at buf[0] (= the reset frame).
                num_frames = 1
                window_start = 0
                window_end = 1
            else:
                window_end = -lag_entries
                window_start = window_end - num_frames
        else:
            window_end = None
            window_start = -num_frames

        videos: dict = {}
        for k, buf in self._frame_buffers.items():
            sliced = buf[window_start:window_end] if window_end is not None else buf[-num_frames:]
            if len(sliced) >= num_frames:
                frames = list(sliced)
            else:
                frames = list(sliced)
                while len(frames) < num_frames:
                    frames.insert(0, sliced[0] if sliced else buf[0])
            videos[k] = np.stack(frames, axis=0) if num_frames > 1 else frames[0]

        # State buffer stores 1 jp/gp per server call (regardless of how many
        # frames were extended). So "lag N chunks" for state = N entries from
        # the end. state_buf[-(lag+1)] gives jp/gp at chunk K-lag's start.
        if window_end is not None:
            jp_send = self._state_buffers["jp"][-(lag + 1)]
            gp_send = self._state_buffers["gp"][-(lag + 1)]
        else:
            jp_send = jp
            gp_send = gp

        return {
            "video.exterior_image_1_left":   videos["video.exterior_image_1_left"],
            "video.exterior_image_2_left":   videos["video.exterior_image_2_left"],
            "video.wrist_image_left":        videos["video.wrist_image_left"],
            "state.joint_position":          jp_send,
            "state.gripper_position":        gp_send,
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
            self._last_pred_composite = None
            self._pending_actions = None
            self._pending_video_pred = None
            self._prev_applied_last = None
            self._prev_applied_full = None
            self._prev_applied_last_gripper = None
            self._two_step_count_since_pause = 0
            self._pending_grip_pause = False
            logger.info("[reset_episode] Cleared _last_video_pred + queue for new episode")

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

        # ASYNC_PRED_AS_OBS=1: 2-step inference per chunk.
        #   A) Run forward on the LAGGED real-obs window — i.e. the same input
        #      sync chunk K-1 would have consumed. With ASYNC_LAG_CHUNKS=1, A_K
        #      mirrors sync chunk K-1 (chunk-1 boundary degrades to 1-frame
        #      buf[0] = reset frame, matching sync chunk 0 exactly). pred_A is
        #      always anchored on real obs so chaining doesn't compound.
        #   B) Replace the entire video time axis with A's full 9-frame decoded
        #      prediction (per-camera split). Re-run forward → pred_B + actions
        #      to apply. Model takes shape[2]==9 directly (skips the 4→9
        #      repeat_interleave path used for 4-frame input).
        # Falls back to single sync forward on chunk 0 (buffer too short to be
        # meaningful — only the reset frame in buf).
        do_async_two_step = (self.ASYNC_PRED_AS_OBS == 1
                             and len(self._frame_buffers["video.exterior_image_1_left"]) > 1)

        two_step_did_run = False
        if do_async_two_step:
            # Step A uses the lagged real-obs window (= policy_obs from the
            # earlier _build_policy_obs(obs) call). At chunk 1 boundary this
            # is 1-frame buf[0]; at K≥2 it's chunk K-1's 4-frame push.
            self._save_truth_video(
                policy_obs, f"pred_{self._infer_count:05d}_I.mp4", label="I"
            )
            result_A, pred_A = self._run_forward(policy_obs)
            pred_A_last = self._save_pred_video_named(
                pred_A, f"pred_{self._infer_count:05d}_A.mp4", label="A"
            )
            actions_A = self._extract_actions(result_A)
            self._save_actions(actions_A, idx=self._infer_count, suffix="_A")
            self._last_pred_composite = pred_A_last  # debug visibility

            # Step B feeds A's full 9-frame decoded prediction (regardless of
            # A's input shape — even when A used 1-frame at chunk 1, its
            # decoded output is still 9 frames, suitable for B's shape[2]==9
            # path).
            policy_obs_B = self._build_b_obs_from_pred(policy_obs, pred_A)
            self._save_truth_video(
                policy_obs_B, f"pred_{self._infer_count:05d}_J.mp4", label="J"
            )
            result_B, pred_B = self._run_forward(policy_obs_B)
            actions_B = self._extract_actions(result_B)
            # Save raw B (model output before shift / amplitude scaling) for
            # offline diagnostics. shift/scale only modify joints 0..6 (gripper
            # is always passed through), so raw B differs from applied B in
            # joint values only.
            self._save_actions(actions_B, idx=self._infer_count, suffix="_B_raw")
            two_step_did_run = True

            # Distill VAE-bias correction: learned residual replaces PD1.
            # Input: (B_raw, B_prev_applied, A_K) channel-concat → (24, 24)
            # Output: residual (24, 7) added to B_raw joints. Set B_PREV_DIFF_ALPHA=0
            # when using distill to avoid double-correction.
            if (self._distill_model is not None
                    and self._prev_applied_full is not None
                    and self._prev_applied_full.shape == actions_B.shape):
                with torch.no_grad():
                    x = np.concatenate(
                        [actions_B[:, :8], self._prev_applied_full[:, :8], actions_A[:, :8]],
                        axis=1
                    ).astype(np.float32)
                    x_t = torch.from_numpy(x).unsqueeze(0)
                    residual = self._distill_model(x_t).squeeze(0).numpy()
                actions_B[:, :7] += residual
                logger.info(
                    f"[distill] residual applied; max|res|={float(np.abs(residual).max()):.4f}"
                )

            # B-action shift correction: B's input is A's video latent round-tripped
            # through VAE.decode → split → resize → VAE.encode. The action head was
            # never trained on this re-encoded distribution, so it produces an OOD
            # per-chunk constant bias (often inducing odd/even oscillation in the
            # executed trajectory). The relative motion inside B is fine, so we
            # translate B's first 7 joints as a whole, preserving B's internal delta.
            # Anchor target: last command we issued at the previous chunk
            # (_prev_applied_last). This is the cleanest async-legal approximation
            # of "where the robot is at chunk K start" — A_K[-1] would also work but
            # is already used by B_PREV_DIFF_ALPHA, so reusing it would double-count
            # A's information. Fall back to A_K[-1] only when no prev-chunk action
            # exists (shouldn't happen for two-step path; defensive).
            if self._prev_applied_last is not None:
                anchor_j = self._prev_applied_last
            else:
                anchor_j = actions_A[-1, :7]
            shift_j = anchor_j - actions_B[0, :7]
            actions_B_shifted = actions_B.copy()
            actions_B_shifted[:, :7] += shift_j

            # B amplitude scaling: B tends to be too flat within a chunk (per-chunk
            # range ~50-80% of A's). Stretch each joint's deviation from B[0] by the
            # ratio of slopes at the K-1/K boundary: scale_j = vA / vB where
            # vA = A[-1,j]-A[-2,j] (A's last-step velocity, end of chunk K-1) and
            # vB = B[1,j]-B[0,j] (B's first-step velocity, start of chunk K). Negative
            # scale means a sign flip; we accept it (per user). Cap the resulting
            # delta from B[0] to A's chunk range so B can't be wilder than A.
            if self.B_AMPLITUDE_SCALE == 1:
                W = max(1, min(self.B_AMPLITUDE_SCALE_WINDOW, actions_A.shape[0] - 1))
                scaled_log = []
                for j in range(7):
                    vA = float(actions_A[-1, j] - actions_A[-1 - W, j])
                    vB = float(actions_B[W, j]   - actions_B[0, j])
                    if abs(vB) < 1e-9:
                        scale_j = 1.0
                    else:
                        scale_j = vA / vB
                    A_range = float(actions_A[:, j].max() - actions_A[:, j].min())
                    delta = (actions_B[:, j] - actions_B[0, j]) * scale_j
                    delta = np.clip(delta, -A_range, A_range)
                    actions_B_shifted[:, j] = actions_B_shifted[0, j] + delta
                    scaled_log.append((scale_j, A_range))
                logger.info(
                    f"[B_amp_scale] W={W} scales={[round(s[0], 2) for s in scaled_log]}, "
                    f"A_ranges={[round(s[1], 3) for s in scaled_log]}"
                )

            # B_PREV_DIFF_ALPHA: residual VAE-bias correction (shape-only).
            # A_K and B_{K-1} both predict chunk K-1's window. Their diff is
            # the VAE round-trip bias observed last chunk. We subtract delta[0]
            # from every step so PD1 contributes 0 at t=0 → anchor=B_{K-1}[-1]
            # is preserved at step 0 (no double-counting of A's information).
            # PD1 only carries the per-step shape difference.
            if (self.B_PREV_DIFF_ALPHA > 0.0
                    and self._prev_applied_full is not None
                    and self._prev_applied_full.shape[0] == actions_A.shape[0]):
                delta = actions_A[:, :7] - self._prev_applied_full[:, :7]
                delta = delta - delta[0:1, :]  # zero PD1 at t=0
                actions_B_shifted[:, :7] += self.B_PREV_DIFF_ALPHA * delta
                logger.info(
                    f"[B_prev_diff] alpha={self.B_PREV_DIFF_ALPHA} (shape-only, t=0 zeroed) "
                    f"delta_max={float(np.abs(delta).max()):.4f} "
                    f"delta_mean_per_j={[round(float(delta[:, j].mean()), 4) for j in range(7)]}"
                )

            # Fallback metric: how badly does A_K's start disagree with the
            # last applied chunk's end? Large value => B's OOD bias likely too
            # severe for shift to fix. In that case, drop B and use A_K
            # directly (lagged real-obs prediction, no round-trip OOD).
            fallback_triggered = False
            fallback_sq = 0.0
            if self._prev_applied_last is not None:
                diff = actions_A[0, :7] - self._prev_applied_last
                fallback_sq = float(np.sum(diff * diff))
                if fallback_sq > self.ASYNC_FALLBACK_SQ_THRESH:
                    fallback_triggered = True

            # FORCE_PAUSE_EVERY_N override: ignore the metric, force pause on a
            # fixed cadence (every N two-step entries since last pause).
            self._two_step_count_since_pause += 1
            forced_pause = (
                self.FORCE_PAUSE_EVERY_N > 0
                and self._two_step_count_since_pause >= self.FORCE_PAUSE_EVERY_N
            )
            # Grip-transition forced pause: previous chunk grasped the object,
            # this chunk is forced to pause to let sim settle + give next chunk
            # fresh post-grasp real obs (chunk-0-like).
            if self._pending_grip_pause:
                forced_pause = True
                self._pending_grip_pause = False
                logger.info("[grip_pause] forcing pause this chunk (gripper closed in previous chunk)")
            # Grip-pre pause: A predicts chunk K-1 with real obs; if A says
            # "should have grasped" anywhere in chunk K-1 but actual previous
            # B (executed) didn't grasp → A is reporting a missed grasp. Pause
            # this chunk and re-plan from fresh obs to prevent error buildup.
            # Use A's MAX gripper across the chunk (not just last) because the
            # model's gripper close signal can be in the middle of the chunk.
            if (self.GRIP_PRE_PAUSE
                    and self._prev_applied_last_gripper is not None
                    and self._prev_applied_last_gripper <= 0.5):
                A_g_max = float(actions_A[:, 7].max())
                if A_g_max > 0.5:
                    forced_pause = True
                    A_g_argmax = int(actions_A[:, 7].argmax())
                    logger.info(
                        f"[grip_pre_pause] A_K says should-have-grasped at step "
                        f"{A_g_argmax} (gripper={A_g_max:.3f}) but prev_applied "
                        f"ended open ({self._prev_applied_last_gripper:.3f}) "
                        f"→ pausing this chunk to fix"
                    )
            # B-abort: if current chunk's B raw plan contains open→close grasp
            # transition, abort this two-step chunk. Grasping should only be
            # performed by post-pause chunk-0-like sync, never by async two-step.
            if (self.GRIP_B_ABORT
                    and float(actions_B[0, 7]) <= 0.5
                    and float(actions_B[-1, 7]) > 0.5):
                forced_pause = True
                logger.info(
                    f"[grip_b_abort] B plans open→close in two-step "
                    f"(B[0,7]={float(actions_B[0,7]):.3f}, "
                    f"B[-1,7]={float(actions_B[-1,7]):.3f}) "
                    f"→ aborting; let next post-pause sync handle grasp"
                )
            if forced_pause:
                fallback_triggered = True

            logger.info(
                f"[async_two_step] step A + step B done; "
                f"shift |max|={float(np.abs(shift_j).max()):.4f}; "
                f"fallback_sq={fallback_sq:.4f} "
                f"(thresh={self.ASYNC_FALLBACK_SQ_THRESH}, triggered={fallback_triggered}"
                f"{', FORCED' if forced_pause else ''})"
            )

            if fallback_triggered:
                # PAUSE: drop both A and B, emit 24 hold-still actions
                # (current jp/gp tiled), and reset server state so the next
                # chunk runs single-frame from the now-static sim. Sim's joint
                # PD controller treats target == current as "stay put"; gripper
                # threshold (>0.2) keeps current open/closed state. KV cache is
                # reset every chunk by FORCE_RESET_EVERY_CHUNK=1, so no extra
                # work needed there.
                hold_jp = obs["observation/joint_position"]
                hold_gp = obs["observation/gripper_position"]
                hold_jp = np.asarray(hold_jp, dtype=np.float32).reshape(-1)[:7]
                hold_gp = np.asarray(hold_gp, dtype=np.float32).reshape(-1)[:1]
                # Snap hold_gp to 0.0 or 1.0 to avoid obs-noise + binarize-threshold
                # interaction. Threshold 0.3 keeps partial grip (obs ~0.35) as
                # "closed" — preserves the held object across pause chunks.
                hold_gp = np.where(hold_gp > 0.3, 1.0, 0.0).astype(np.float32)
                hold_step = np.concatenate([hold_jp, hold_gp])  # (8,)
                H_actions = actions_A.shape[0] if actions_A is not None else 24
                actions_override = np.tile(hold_step[None, :], (H_actions, 1)).astype(np.float32)
                # Buffers / queue / pred-cache cleared; pending_reset makes the
                # next chunk's _build_policy_obs behave like reset_episode=True.
                for k in self._frame_buffers:
                    self._frame_buffers[k].clear()
                for k in self._state_buffers:
                    self._state_buffers[k].clear()
                self._is_first_call = True
                self._last_pred_composite = None
                self._last_video_pred = None
                self._pending_actions = None
                self._pending_video_pred = None
                self._pending_reset = True
                self._two_step_count_since_pause = 0
                # _prev_applied_last will be re-set below from the hold-still actions
                # so the next chunk's fallback metric compares against the static pose.
                result_batch = result_B
                # Keep B's prediction video for *saving* (diagnostics) but DON'T
                # carry it forward as next-chunk's video latent context.
                video_pred = pred_B
                video_pred_no_carry = True
                logger.info(
                    f"[async_two_step] FALLBACK PAUSE: emitting {H_actions} hold-still "
                    f"actions (jp={hold_jp.tolist()}, gp={float(hold_gp[0]):.3f}); "
                    f"buffers cleared, next chunk = fresh chunk-0"
                )
            elif self.ASYNC_QUEUE == 1:
                if self._pending_actions is not None:
                    actions_override = self._pending_actions
                    video_pred_override = self._pending_video_pred
                    logger.info("[async_queue] applied queued B from previous chunk")
                else:
                    actions_override = actions_A
                    video_pred_override = pred_A
                    logger.info("[async_queue] no pending yet → using A this chunk")
                self._pending_actions = actions_B_shifted
                self._pending_video_pred = pred_B
                result_batch = result_B
                video_pred = video_pred_override
            else:
                # Legacy QUEUE=0 + shift correction.
                actions_override = actions_B_shifted
                result_batch = result_B
                video_pred = pred_B
        elif predicted_mode and self._last_video_pred is not None:
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
            self._save_truth_video(
                policy_obs, f"pred_{self._infer_count:05d}_I.mp4", label="I"
            )
            result_batch, video_pred = self._run_forward(policy_obs, latent_video=lv)
        else:
            # === Sync mode (default): use real obs ===
            self._save_truth_video(
                policy_obs, f"pred_{self._infer_count:05d}_I.mp4", label="I"
            )
            result_batch, video_pred = self._run_forward(policy_obs)

        # If two_step ran, actions_override holds the to-apply commands
        # (shifted B for QUEUE=0, queued shifted B / A_K fallback for QUEUE=1).
        # Otherwise extract raw from result_batch.
        if two_step_did_run and 'actions_override' in locals() and actions_override is not None:
            actions = actions_override
        else:
            actions = self._extract_actions(result_batch)

        if (self.RELATIVE_ACTION_SCALE != 1.0
                and actions is not None and actions.shape[-1] >= 7):
            current_jp = np.asarray(policy_obs["state.joint_position"], dtype=actions.dtype).reshape(1, -1)[:, :7]
            actions[:, :7] = current_jp + self.RELATIVE_ACTION_SCALE * (actions[:, :7] - current_jp)

        # Update _prev_applied_last for next chunk's fallback metric and
        # _prev_applied_full for B_PREV_DIFF_ALPHA correction. When pause is
        # pending (next chunk = chunk-0-like), clear full so the diff doesn't
        # carry across a discontinuity.
        if actions is not None and actions.shape[-1] >= 7:
            self._prev_applied_last = actions[-1, :7].copy()
            if actions.shape[-1] >= 8:
                self._prev_applied_last_gripper = float(actions[-1, 7])
            if self._pending_reset:
                self._prev_applied_full = None
            else:
                self._prev_applied_full = actions.copy()

        # Detect open→closed gripper transition within this chunk. If the
        # gripper command starts open (<=0.2) and ends closed (>0.2), the
        # grasp happens during this chunk → force a pause on the next chunk.
        # Skip if this chunk was a fallback hold-still (gripper constant).
        if (self.GRIP_TRANSITION_PAUSE
                and actions is not None and actions.shape[-1] >= 8
                and not self._pending_reset):
            g_first = float(actions[0, 7])
            g_last = float(actions[-1, 7])
            if g_first <= 0.5 < g_last:
                self._pending_grip_pause = True
                logger.info(
                    f"[grip_pause] open→closed detected this chunk "
                    f"(first={g_first:.3f}, last={g_last:.3f}); "
                    f"next chunk will be forced to pause"
                )

        t_forward = time.perf_counter()

        # Save video_pred for next chunk's latent context.
        # On fallback pause, video_pred_no_carry was set so we keep `video_pred`
        # for *saving* (diagnostics) but don't carry it across the pause.
        if 'video_pred_no_carry' in locals() and video_pred_no_carry:
            self._last_video_pred = None
        else:
            self._last_video_pred = video_pred

        # Decode and save AI prediction video. In 2-step async mode, save the
        # final (B) under the canonical pred_NNNNN.mp4 name with a "B" label;
        # pred_A was already saved above with an "A" label. Otherwise use the
        # standard unlabeled save.
        if two_step_did_run:
            self._save_pred_video_named(
                video_pred, f"pred_{self._infer_count:05d}.mp4", label="B"
            )
            self._infer_count += 1  # mirror _save_pred_video's auto-increment
        else:
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

            # Save LAST frame of the decoded pred composite for ASYNC_PRED_AS_OBS.
            # Layout: top half = wrist (2x wide), bottom-left = left exterior,
            # bottom-right = right exterior (per dreamzero_cotrain._prepare_video).
            self._last_pred_composite = frames[-1].copy()

            out_path = os.path.join(self.video_out_dir, f"pred_{self._infer_count:05d}.mp4")
            imageio.mimsave(out_path, list(frames), fps=15, codec="libx264")
            logger.info(f"[pred video] saved {len(frames)} frames → {out_path}")
        except Exception as e:
            logger.warning(f"[pred video] failed to save: {e}")
        finally:
            self._infer_count += 1

    def _save_actions(self, actions: np.ndarray, idx: int = None, suffix: str = "") -> None:
        """Save full action array (H, 8) as .npy and human-readable .txt.

        Args:
            idx: chunk index. If None, uses self._infer_count - 1 (legacy: assumes
                _save_pred_video already incremented).
            suffix: filename suffix (e.g. '_A' for forward A's action set).
        """
        if idx is None:
            idx = self._infer_count - 1
        try:
            npy_path = os.path.join(self.video_out_dir, f"pred_{idx:05d}{suffix}_actions.npy")
            txt_path = os.path.join(self.video_out_dir, f"pred_{idx:05d}{suffix}_actions.txt")

            np.save(npy_path, actions)

            # Human-readable: header + rows
            H = actions.shape[0]
            with open(txt_path, "w") as f:
                f.write(f"# chunk {idx:05d}{suffix}  shape=({H}, 8)\n")
                f.write(f"# cols: j0  j1  j2  j3  j4  j5  j6  gripper\n")
                for step, row in enumerate(actions):
                    vals = "  ".join(f"{v:+.4f}" for v in row)
                    f.write(f"step {step:02d}: {vals}\n")

            # Also log gripper column to server log for quick inspection
            gripper_vals = actions[:, -1]
            logger.info(
                f"[actions {idx:05d}{suffix}] gripper={np.array2string(gripper_vals, precision=3, separator=',')}  "
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
