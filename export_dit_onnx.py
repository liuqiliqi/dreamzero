#!/usr/bin/env python3
"""Export the DreamZero DiT diffusion backbone to ONNX.

This script wraps ``CausalWanModel._forward_inference_trt_droid`` in a thin
``nn.Module`` so that all inputs are positional tensors (no Optional, no
lists), making the graph ONNX-exportable.

Usage
-----
    # Must set ENABLE_TENSORRT=true so the model uses ONNX-compatible ops
    ENABLE_TENSORRT=true python export_dit_onnx.py \
        --weights /fact_data/qiliu/dreamzero_weights/DreamZero-DROID \
        --output dit_droid.onnx \
        --kv-len 1320

Environment variables
---------------------
    ENABLE_TENSORRT=true   (required) – switches attention to SDPA / non-polar RoPE
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DROID model constants
# ---------------------------------------------------------------------------
DROID_CONFIG = dict(
    dim=5120,
    ffn_dim=13824,
    num_heads=40,
    num_layers=40,
    head_dim=128,
    x_channels=16,   # out_dim: noisy video latent channels (before y concat)
    y_channels=20,   # conditional video channels (concat inside _forward_inference)
    in_dim=36,       # x_channels + y_channels → patch_embedding input
    out_dim=16,
    text_len=512,
    text_dim=4096,
    frame_seqlen=880,  # per-frame tokens after patching: (44//2)*(80//2) = 22*40 = 880
    num_frame_per_block=2,
    action_horizon=24,
    action_dim=32,
    clip_dim=1280,
    clip_len=257,
    state_dim=64,
    # Latent spatial dims: video 352x640 → VAE(÷8) → 44x80
    H_lat=44,
    W_lat=80,
)


class TRTExportWrapper(nn.Module):
    """Thin wrapper that calls ``_forward_inference`` directly with correct
    parameters, bypassing ``_forward_inference_trt_droid`` (which has a
    hardcoded seq_len bug).

    All inputs are required tensors (no Optional, no lists) so that
    ``torch.onnx.export`` can trace the graph cleanly.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        kv_cache: torch.Tensor,
        y: torch.Tensor,
        clip_feature: torch.Tensor,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute seq_len from the actual input geometry:
        # After patch_embedding (stride 1,2,2): spatial dims halved.
        # x shape: [B, C, F, H, W] → after patching: [B, dim, F, H//2, W//2]
        # seq_len = F * (H//2) * (W//2)
        F = x.shape[2]                     # num_frame_per_block (2)
        H_lat, W_lat = x.shape[3], x.shape[4]  # latent spatial (22, 40)
        per_frame_tokens = (H_lat // 2) * (W_lat // 2)  # 11 * 20 = 220
        seq_len = per_frame_tokens * F     # 440

        # Unpack KV cache: [num_layers, 2, B, kv_len, heads, head_dim] → list
        kv_cache_list = []
        for block_index in range(len(self.model.blocks)):
            kv_cache_list.append(kv_cache[block_index])

        # Compute current_start_frame from KV cache length.
        # Each frame contributes per_frame_tokens to the cache (action/state
        # tokens are appended separately and are NOT part of kv_len for
        # frame counting).
        kv_len = kv_cache.shape[3]
        current_start_frame = kv_len // per_frame_tokens

        video_noise_pred, action_noise_pred, _ = self.model._forward_inference(
            x=x,
            timestep=timestep,
            context=context,
            seq_len=seq_len,
            kv_cache=kv_cache_list,
            crossattn_cache=None,
            y=y,
            clip_feature=clip_feature,
            action=action,
            timestep_action=timestep_action,
            state=state,
            current_start_frame=current_start_frame,
        )
        # Ensure action_noise_pred is never None (ONNX requires concrete tensor)
        if action_noise_pred is None:
            action_noise_pred = torch.zeros(1, device=x.device, dtype=x.dtype)
        return video_noise_pred, action_noise_pred


def _load_model(weights_path: str, device: torch.device) -> nn.Module:
    """Load the CausalWanModel (DiT) from a DreamZero VLA checkpoint.

    The VLA checkpoint stores DiT weights under the ``action_head.model.*``
    prefix.  We instantiate CausalWanModel from the config and load the
    stripped state dict.
    """
    import json
    from safetensors.torch import load_file

    # Add project root to path so imports resolve
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanModel,
    )

    weights_path = os.path.abspath(weights_path)

    # Read the VLA config to extract DiT constructor kwargs
    config_path = os.path.join(weights_path, "config.json")
    with open(config_path) as f:
        vla_config = json.load(f)
    dit_cfg = vla_config["action_head_cfg"]["config"]["diffusion_model_cfg"]
    # Remove hydra keys
    dit_kwargs = {k: v for k, v in dit_cfg.items() if not k.startswith("_")}
    logger.info("Instantiating CausalWanModel with config: %s", dit_kwargs)
    model = CausalWanModel(**dit_kwargs)

    # Load weights from sharded safetensors
    index_path = os.path.join(weights_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    # Collect DiT weights (prefix: action_head.model.)
    dit_prefix = "action_head.model."
    shard_files = set()
    for key, shard_file in index["weight_map"].items():
        if key.startswith(dit_prefix):
            shard_files.add(shard_file)

    state_dict = {}
    for shard_file in sorted(shard_files):
        shard_path = os.path.join(weights_path, shard_file)
        logger.info("Loading shard: %s", shard_path)
        shard_data = load_file(shard_path)
        for key, value in shard_data.items():
            if key.startswith(dit_prefix):
                stripped_key = key[len(dit_prefix):]
                state_dict[stripped_key] = value

    logger.info("Loaded %d DiT weight tensors", len(state_dict))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    logger.info("Model loaded successfully (%d parameters)", sum(p.numel() for p in model.parameters()))
    return model


def _create_dummy_inputs(
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Create dummy tensors matching the DROID model's expected shapes."""
    C = DROID_CONFIG
    H_lat, W_lat = C["H_lat"], C["W_lat"]
    return dict(
        x=torch.randn(1, C["x_channels"], C["num_frame_per_block"], H_lat, W_lat, device=device, dtype=dtype),
        timestep=torch.randint(0, 1000, (1, C["num_frame_per_block"]), device=device, dtype=torch.int64),
        context=torch.randn(1, C["text_len"], C["text_dim"], device=device, dtype=dtype),
        kv_cache=torch.randn(
            C["num_layers"], 2, 1, kv_len, C["num_heads"], C["head_dim"],
            device=device, dtype=dtype,
        ),
        y=torch.randn(1, C["y_channels"], C["num_frame_per_block"], H_lat, W_lat, device=device, dtype=dtype),
        clip_feature=torch.randn(1, C["clip_len"], C["clip_dim"], device=device, dtype=dtype),
        action=torch.randn(1, C["action_horizon"], C["action_dim"], device=device, dtype=dtype),
        timestep_action=torch.randint(0, 1000, (1, C["action_horizon"]), device=device, dtype=torch.int64),
        state=torch.randn(1, 1, C["state_dim"], device=device, dtype=dtype),
    )


def export_onnx(
    weights_path: str,
    output_path: str,
    kv_len: int = 1320,
    opset_version: int = 17,
    use_dynamo: bool = False,
) -> None:
    """Export the DiT model to ONNX format.

    Parameters
    ----------
    weights_path : str
        Path to pretrained CausalWanModel weights.
    output_path : str
        Where to write the ``.onnx`` file.
    kv_len : int
        KV cache sequence length for the dummy inputs (multiples of
        frame_seqlen=440).  Default 1320 = 3 chunks.
    opset_version : int
        ONNX opset version (17 recommended for TRT 10+).
    use_dynamo : bool
        If True, use ``torch.onnx.export(dynamo=True)`` (TorchDynamo-based
        exporter) as a fallback for tricky ops.
    """
    device = torch.device("cuda:0")

    # Load model
    model = _load_model(weights_path, device)
    wrapper = TRTExportWrapper(model)
    wrapper.eval()

    # Create dummy inputs
    dummy = _create_dummy_inputs(kv_len, device)
    input_names = list(dummy.keys())
    output_names = ["video_noise_pred", "action_noise_pred"]

    # Dynamic axes for kv_cache dim 3 (kv_len)
    dynamic_axes = {
        "kv_cache": {3: "kv_len"},
    }

    args = tuple(dummy[name] for name in input_names)

    logger.info("Starting ONNX export (opset=%d, kv_len=%d, dynamo=%s)...", opset_version, kv_len, use_dynamo)

    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    # Export to a temporary path first, then re-save with external data
    # for models >2GB.
    tmp_path = output_path + ".tmp"

    if use_dynamo:
        # TorchDynamo-based exporter — better for complex control flow
        export_output = torch.onnx.export(
            wrapper,
            args,
            dynamo=True,
        )
        export_output.save(tmp_path)
    else:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                args,
                tmp_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
            )

    # Re-save with external data so protobuf stays under 2GB
    logger.info("Converting to external data format for large model...")
    onnx_model = onnx.load(tmp_path)
    data_filename = os.path.basename(output_path) + ".data"
    convert_model_to_external_data(
        onnx_model,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save_model(onnx_model, output_path)
    # Remove temporary file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    logger.info("ONNX model exported to %s (external data: %s)", output_path, data_filename)

    # Quick validation
    try:
        onnx.checker.check_model(output_path)
        logger.info("ONNX model validation passed")
    except Exception as e:
        logger.error("ONNX validation failed: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DreamZero DiT to ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default="/fact_data/qiliu/dreamzero_weights/DreamZero-DROID",
        help="Path to pretrained model weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dit_droid.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--kv-len",
        type=int,
        default=1320,
        help="KV cache sequence length (must be multiple of 440)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Use torch.onnx.export(dynamo=True) for complex ops",
    )
    args = parser.parse_args()

    # Validate env
    if os.getenv("ENABLE_TENSORRT", "").lower() != "true":
        logger.warning(
            "ENABLE_TENSORRT is not set to 'true'. The export may fail due to "
            "non-ONNX-compatible ops (flash attention, polar RoPE). "
            "Set ENABLE_TENSORRT=true for best results."
        )

    export_onnx(
        weights_path=args.weights,
        output_path=args.output,
        kv_len=args.kv_len,
        opset_version=args.opset,
        use_dynamo=args.dynamo,
    )


if __name__ == "__main__":
    main()
