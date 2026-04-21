"""TensorRT engine wrapper for DreamZero DiT inference.

Provides a TRTEngine class that deserializes a TensorRT engine, manages GPU
buffers, and exposes a __call__ interface matching the signature expected by
_run_diffusion_steps() in wan_flow_matching_action_tf.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import tensorrt as trt

logger = logging.getLogger(__name__)

# TRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ---------------------------------------------------------------------------
# Mapping between torch dtypes and TRT / numpy dtypes
# ---------------------------------------------------------------------------

_TORCH_TO_NP: dict[torch.dtype, np.dtype] = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.bfloat16: np.float16,  # TRT doesn't natively bind bf16 via numpy; cast to fp16
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.bool: np.bool_,
}

_TRT_TO_TORCH: dict[trt.DataType, torch.dtype] = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.FP8: torch.float8_e4m3fn,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
}


class TRTEngine:
    """Wraps a serialized TensorRT engine for inference.

    Parameters
    ----------
    engine_path : str | Path
        Path to the serialized ``.engine`` (or ``.plan``) file.
    device : torch.device | int
        CUDA device to use.  Defaults to ``cuda:0``.
    """

    def __init__(
        self,
        engine_path: str | Path,
        device: torch.device | int = 0,
    ) -> None:
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # Resolve bare "cuda" to the current device index
        if device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.device = device

        # Ensure we're on the correct CUDA device before deserializing
        torch.cuda.set_device(self.device)

        # Deserialize the engine
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TRT execution context")

        # Create a dedicated CUDA stream
        self.stream = torch.cuda.Stream(device=self.device)

        # Discover input / output tensor names and their properties
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.output_dtypes: dict[str, torch.dtype] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                trt_dtype = self.engine.get_tensor_dtype(name)
                self.output_dtypes[name] = _TRT_TO_TORCH.get(trt_dtype, torch.float32)

        logger.info(
            "TRT engine loaded: %d inputs (%s), %d outputs (%s)",
            len(self.input_names),
            self.input_names,
            len(self.output_names),
            self.output_names,
        )

        # Pre-allocated output buffers (populated on first call or shape change)
        self._output_buffers: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_contiguous(self, t: torch.Tensor) -> torch.Tensor:
        """Return a contiguous tensor on the correct device, casting bf16→fp16
        if the engine was built with fp16 weights."""
        if t.device != self.device:
            t = t.to(device=self.device)
        if not t.is_contiguous():
            t = t.contiguous()
        return t

    def _allocate_output(self, name: str) -> torch.Tensor:
        """Allocate (or re-use) an output buffer based on the execution context's
        current shape for *name*."""
        shape = tuple(self.context.get_tensor_shape(name))
        dtype = self.output_dtypes[name]
        existing = self._output_buffers.get(name)
        if existing is not None and existing.shape == shape and existing.dtype == dtype:
            return existing
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        self._output_buffers[name] = buf
        return buf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        *,
        context: torch.Tensor,
        kv_cache: torch.Tensor,
        y: torch.Tensor,
        clip_feature: torch.Tensor,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the TRT engine with the same interface as
        ``CausalWanModel._forward_inference_trt_droid``.

        Returns
        -------
        obs_noise_pred : Tensor
            Video noise prediction.
        action_noise_pred : Tensor
            Action noise prediction.
        """
        # Map keyword arguments to the ONNX input names (positional order used
        # during export).
        inputs_by_name: dict[str, torch.Tensor] = {
            "x": x,
            "timestep": timestep,
            "context": context,
            "kv_cache": kv_cache,
            "y": y,
            "clip_feature": clip_feature,
            "action": action,
            "timestep_action": timestep_action,
            "state": state,
        }

        with torch.cuda.stream(self.stream):
            # Bind inputs
            for name in self.input_names:
                tensor = self._ensure_contiguous(inputs_by_name[name])
                self.context.set_input_shape(name, tuple(tensor.shape))
                self.context.set_tensor_address(name, tensor.data_ptr())

            # Allocate & bind outputs
            outputs: dict[str, torch.Tensor] = {}
            for name in self.output_names:
                buf = self._allocate_output(name)
                self.context.set_tensor_address(name, buf.data_ptr())
                outputs[name] = buf

            # Execute
            ok = self.context.execute_async_v3(self.stream.cuda_stream)
            if not ok:
                raise RuntimeError("TRT execute_async_v3 failed")

        # Synchronize before returning so the caller can read results
        self.stream.synchronize()

        # Return in the order expected by _run_diffusion_steps:
        # (obs_noise_pred, action_noise_pred)
        obs_noise_pred = outputs["video_noise_pred"]
        action_noise_pred = outputs["action_noise_pred"]

        # Cast back to bf16 if the engine produced fp16
        if obs_noise_pred.dtype != torch.bfloat16:
            obs_noise_pred = obs_noise_pred.to(torch.bfloat16)
        if action_noise_pred.dtype != torch.bfloat16:
            action_noise_pred = action_noise_pred.to(torch.bfloat16)

        return obs_noise_pred, action_noise_pred


def load_tensorrt_engine(
    engine_path: str | Path,
    model_type: str = "ar_14B",
    device: torch.device | int = 0,
) -> TRTEngine:
    """Factory function to load a TensorRT engine.

    Parameters
    ----------
    engine_path : str or Path
        Path to the serialized engine file.
    model_type : str
        Model identifier (for future multi-model support). Currently only
        ``"ar_14B"`` is used.
    device : torch.device or int
        CUDA device ordinal.

    Returns
    -------
    TRTEngine
    """
    logger.info("Loading TRT engine for model_type=%s from %s", model_type, engine_path)
    engine = TRTEngine(engine_path, device=device)
    logger.info("TRT engine ready (%d inputs, %d outputs)", len(engine.input_names), len(engine.output_names))
    return engine
