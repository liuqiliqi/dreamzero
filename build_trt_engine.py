#!/usr/bin/env python3
"""Build a TensorRT engine from the exported ONNX model.

Usage
-----
    python build_trt_engine.py \
        --onnx dit_droid.onnx \
        --output dit_droid.engine \
        --bf16 \
        --workspace 8

The resulting ``.engine`` file can be loaded at runtime via::

    LOAD_TRT_ENGINE=dit_droid.engine python <your_inference_script>
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import tensorrt as trt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

# ---------------------------------------------------------------------------
# DROID model constants for optimization profiles
# ---------------------------------------------------------------------------
# KV cache shape: [num_layers=40, 2, batch=1, kv_len, num_heads=40, head_dim=128]
# kv_len varies: min=880 (1 chunk), opt=2640 (3 chunks), max=8800 (10 chunks)
# per_frame_tokens = (44//2)*(80//2) = 22*40 = 880, kv_len = 880 * num_cached_frames
# Observed growth: 880 → 2640 → 4400 → 6160 (step=1760 per KV prefill cycle)
KV_CACHE_PROFILE = {
    "min": (40, 2, 1, 880, 40, 128),
    "opt": (40, 2, 1, 2640, 40, 128),
    "max": (40, 2, 1, 8800, 40, 128),
}

# All other inputs have fixed shapes for B=1 DROID
# Latent spatial: video 352x640 → VAE(÷8) → 44x80
FIXED_INPUTS = {
    "x":                {"shape": (1, 16, 2, 44, 80)},
    "timestep":         {"shape": (1, 2)},
    "context":          {"shape": (1, 512, 4096)},
    "y":                {"shape": (1, 20, 2, 44, 80)},
    "clip_feature":     {"shape": (1, 257, 1280)},
    "action":           {"shape": (1, 24, 32)},
    "timestep_action":  {"shape": (1, 24)},
    "state":            {"shape": (1, 1, 64)},
}


def build_engine(
    onnx_path: str,
    output_path: str,
    workspace_gb: int = 8,
    bf16: bool = True,
    fp16: bool = False,
    fp8: bool = False,
    verbose: bool = False,
) -> None:
    """Parse ONNX and build a TensorRT engine.

    Parameters
    ----------
    onnx_path : str
        Path to the exported ``.onnx`` file.
    output_path : str
        Where to write the serialized engine.
    workspace_gb : int
        GPU workspace memory limit in GiB.
    bf16 : bool
        Enable BF16 precision (recommended for H20).
    fp16 : bool
        Enable FP16 precision.
    fp8 : bool
        Enable FP8 precision (for Q/DQ-annotated ONNX models from modelopt).
    verbose : bool
        Turn on TRT verbose logging.
    """
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if verbose:
        TRT_LOGGER.min_severity = trt.Logger.VERBOSE

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # Precision flags
    if bf16:
        # TRT 10.x: BF16 flag exists but no platform_has_fast_bf16 query.
        # H20 supports BF16 natively; just set the flag.
        try:
            config.set_flag(trt.BuilderFlag.BF16)
            logger.info("BF16 precision enabled")
        except AttributeError:
            logger.warning("BF16 flag not available in this TRT version, using FP16")
            config.set_flag(trt.BuilderFlag.FP16)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 precision enabled")
    if fp8:
        try:
            config.set_flag(trt.BuilderFlag.FP8)
            logger.info("FP8 precision enabled (for Q/DQ-annotated layers)")
        except AttributeError:
            logger.warning("FP8 flag not available in this TRT version")

    # Workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    logger.info("Workspace limit: %d GiB", workspace_gb)

    # Parse ONNX
    logger.info("Parsing ONNX model from %s ...", onnx_path)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("ONNX parse error %d: %s", i, parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")
    logger.info("ONNX parsed successfully: %d inputs, %d outputs",
                network.num_inputs, network.num_outputs)

    # Log network I/O
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        logger.info("  Input  %d: name=%s shape=%s dtype=%s", i, inp.name, inp.shape, inp.dtype)
    for i in range(network.num_outputs):
        out = network.get_output(i)
        logger.info("  Output %d: name=%s shape=%s dtype=%s", i, out.name, out.shape, out.dtype)

    # Optimization profile for dynamic kv_len
    profile = builder.create_optimization_profile()

    for name, info in FIXED_INPUTS.items():
        shape = info["shape"]
        profile.set_shape(name, shape, shape, shape)

    profile.set_shape(
        "kv_cache",
        KV_CACHE_PROFILE["min"],
        KV_CACHE_PROFILE["opt"],
        KV_CACHE_PROFILE["max"],
    )

    config.add_optimization_profile(profile)
    logger.info("Optimization profile added (kv_len: min=%d, opt=%d, max=%d)",
                KV_CACHE_PROFILE["min"][3], KV_CACHE_PROFILE["opt"][3], KV_CACHE_PROFILE["max"][3])

    # Build engine
    logger.info("Building TensorRT engine (this may take a while)...")
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed")
    elapsed = time.time() - t0

    # IHostMemory uses .nbytes for size
    engine_bytes = bytes(serialized_engine)
    logger.info("Engine built in %.1f seconds (%.1f MiB)",
                elapsed, len(engine_bytes) / (1024 * 1024))

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(engine_bytes)
    logger.info("Engine saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TRT engine from ONNX")
    parser.add_argument("--onnx", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--output", type=str, default="dit_droid.engine", help="Output engine path")
    parser.add_argument("--workspace", type=int, default=8, help="Workspace memory in GiB")
    parser.add_argument("--bf16", action="store_true", default=True, help="Enable BF16 (default)")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16")
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 (for Q/DQ-annotated ONNX from modelopt)")
    parser.add_argument("--no-bf16", action="store_true", help="Disable BF16")
    parser.add_argument("--verbose", action="store_true", help="Verbose TRT logging")
    args = parser.parse_args()

    bf16 = args.bf16 and not args.no_bf16

    build_engine(
        onnx_path=args.onnx,
        output_path=args.output,
        workspace_gb=args.workspace,
        bf16=bf16,
        fp16=args.fp16,
        fp8=args.fp8,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
