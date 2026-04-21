#!/usr/bin/env python3
"""Quantize the DreamZero DiT to FP8 and export to ONNX.

Uses nvidia-modelopt (mtq) for post-training quantization with FP8.
The quantized model includes Q/DQ nodes that TensorRT uses for FP8 inference.

Usage
-----
    ENABLE_TENSORRT=true python quantize_dit_fp8.py \
        --weights /fact_data/qiliu/dreamzero_weights/DreamZero-DROID \
        --output dit_droid_fp8.onnx \
        --calib-steps 8

Environment variables
---------------------
    ENABLE_TENSORRT=true   (required) – switches attention to SDPA / non-polar RoPE
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Reuse model loading and config from export_dit_onnx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_dit_onnx import (
    DROID_CONFIG,
    TRTExportWrapper,
    _create_dummy_inputs,
    _load_model,
)


def _build_quant_config():
    """Build FP8 quantization config, skipping embeddings/norms/head."""
    import modelopt.torch.quantization as mtq

    quant_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    # Skip sensitive layers for accuracy
    for pattern in [
        "*patch_embedding*",
        "*text_embedding*",
        "*time_embedding*",
        "*head*",
        "*norm*",
    ]:
        quant_cfg["quant_cfg"][pattern] = {"enable": False}
    return quant_cfg


def _make_calib_loop(
    wrapper: nn.Module,
    device: torch.device,
    kv_lens: list[int],
    calib_steps: int,
):
    """Return a calibration forward_loop function for mtq.quantize().

    mtq.quantize calls forward_loop(model) — the function must run
    representative forward passes so the quantizer can collect activation
    statistics.
    """

    def forward_loop(model):
        model.eval()
        with torch.no_grad():
            for step in range(calib_steps):
                kv_len = kv_lens[step % len(kv_lens)]
                dummy = _create_dummy_inputs(kv_len, device)
                args = tuple(dummy.values())
                logger.info(
                    "  Calibration step %d/%d (kv_len=%d)", step + 1, calib_steps, kv_len
                )
                model(*args)

    return forward_loop


def quantize_and_export(
    weights_path: str,
    output_path: str,
    kv_len_export: int = 1320,
    kv_lens_calib: list[int] | None = None,
    calib_steps: int = 8,
    opset_version: int = 17,
) -> None:
    """Quantize DiT to FP8 and export ONNX with Q/DQ nodes.

    Parameters
    ----------
    weights_path : str
        Path to pretrained model weights.
    output_path : str
        Where to write the FP8 ONNX file.
    kv_len_export : int
        KV cache length for the ONNX export trace.
    kv_lens_calib : list[int]
        KV cache lengths to cycle through during calibration.
    calib_steps : int
        Number of calibration forward passes.
    opset_version : int
        ONNX opset version.
    """
    import modelopt.torch.quantization as mtq

    if kv_lens_calib is None:
        kv_lens_calib = [880, 2640, 4400]

    device = torch.device("cuda:0")

    # 1. Load model
    logger.info("Loading model from %s", weights_path)
    model = _load_model(weights_path, device)
    wrapper = TRTExportWrapper(model)
    wrapper.eval()

    # 2. Quantize with FP8
    logger.info("Quantizing model with FP8 (calib_steps=%d, kv_lens=%s)", calib_steps, kv_lens_calib)
    quant_cfg = _build_quant_config()
    calib_loop = _make_calib_loop(wrapper, device, kv_lens_calib, calib_steps)

    mtq.quantize(wrapper, quant_cfg, forward_loop=calib_loop)
    logger.info("FP8 quantization complete")

    # 3. Export to ONNX — modelopt registers custom symbolic functions
    #    that emit Q/DQ nodes automatically
    dummy = _create_dummy_inputs(kv_len_export, device)
    input_names = list(dummy.keys())
    output_names = ["video_noise_pred", "action_noise_pred"]
    dynamic_axes = {"kv_cache": {3: "kv_len"}}
    args = tuple(dummy[name] for name in input_names)

    logger.info("Exporting quantized ONNX (opset=%d, kv_len=%d)...", opset_version, kv_len_export)

    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    tmp_path = output_path + ".tmp"

    # modelopt patches torch.onnx.export to handle Q/DQ nodes
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

    # Re-save with external data for large model
    logger.info("Converting to external data format...")
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
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    logger.info("FP8 ONNX model exported to %s (external data: %s)", output_path, data_filename)

    # Validate
    try:
        onnx.checker.check_model(output_path)
        logger.info("ONNX model validation passed")
    except Exception as e:
        logger.warning("ONNX validation warning: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize DreamZero DiT to FP8 and export ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default="/fact_data/qiliu/dreamzero_weights/DreamZero-DROID",
        help="Path to pretrained model weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dit_droid_fp8.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--kv-len",
        type=int,
        default=1320,
        help="KV cache length for ONNX export trace",
    )
    parser.add_argument(
        "--calib-kv-lens",
        type=int,
        nargs="+",
        default=[880, 2640, 4400],
        help="KV cache lengths for calibration",
    )
    parser.add_argument(
        "--calib-steps",
        type=int,
        default=8,
        help="Number of calibration forward passes",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    args = parser.parse_args()

    if os.getenv("ENABLE_TENSORRT", "").lower() != "true":
        logger.warning(
            "ENABLE_TENSORRT is not set to 'true'. The export may fail due to "
            "non-ONNX-compatible ops. Set ENABLE_TENSORRT=true."
        )

    quantize_and_export(
        weights_path=args.weights,
        output_path=args.output,
        kv_len_export=args.kv_len,
        kv_lens_calib=args.calib_kv_lens,
        calib_steps=args.calib_steps,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
