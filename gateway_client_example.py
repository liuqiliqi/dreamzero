#!/usr/bin/env python3
"""Example client showing how external devices interact with the DreamZero gateway.

This demonstrates two modes:
    1. Single-frame mode: send 1 frame at a time via /predict (gateway buffers automatically)
    2. Batch mode: send multiple frames at once via /predict_batch

Usage:
    # With real camera frames (from video files):
    python gateway_client_example.py --gateway http://localhost:8080

    # With zero/dummy frames:
    python gateway_client_example.py --gateway http://localhost:8080 --use-zero-images
"""

import argparse
import json
import logging
import os
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "debug_image")
CAMERA_FILES = {
    "exterior_0": "exterior_image_1_left.mp4",
    "exterior_1": "exterior_image_2_left.mp4",
    "wrist": "wrist_image_left.mp4",
}


def post_json(url: str, data: dict) -> dict:
    """Send a JSON POST request and return the parsed response."""
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def load_video_frames(path: str) -> np.ndarray:
    """Load all frames from a video. Returns (N, H, W, 3) uint8 RGB."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def demo_single_frame_mode(gateway_url: str, prompt: str, use_zero: bool):
    """Send frames one at a time. Gateway buffers and infers when ready.

    Flow:
        Step 0 (initial): send 1 frame  → get actions (24, 8)
        Step 1: send 4 frames one by one → get actions on the 4th frame
        Step 2: send 4 frames one by one → get actions on the 4th frame
        ...
    """
    logger.info("=== Single-frame mode ===")

    # Start session
    resp = post_json(f"{gateway_url}/start", {"prompt": prompt})
    session_id = resp["session_id"]
    logger.info(f"Session started: {session_id}")

    if use_zero:
        h, w = 180, 320
        make_frame = lambda: np.zeros((h, w, 3), dtype=np.uint8)
        total_frames = 50
    else:
        cameras = {}
        for cam, fname in CAMERA_FILES.items():
            cameras[cam] = load_video_frames(os.path.join(VIDEO_DIR, fname))
            logger.info(f"  Loaded {cam}: {cameras[cam].shape}")
        total_frames = min(v.shape[0] for v in cameras.values())

    num_inferences = 0
    for i in range(min(total_frames, 50)):
        if use_zero:
            ext0 = ext1 = wrist = make_frame()
        else:
            ext0 = cameras["exterior_0"][i]
            ext1 = cameras["exterior_1"][i]
            wrist = cameras["wrist"][i]

        t0 = time.time()
        resp = post_json(f"{gateway_url}/predict", {
            "session_id": session_id,
            "exterior_0": ext0.tolist(),
            "exterior_1": ext1.tolist(),
            "wrist": wrist.tolist(),
        })
        dt = time.time() - t0

        if resp["status"] == "buffering":
            logger.info(f"  Frame {i}: buffering ({resp['buffered']}/{resp['needed']})")
        else:
            num_inferences += 1
            actions = np.array(resp["actions"])
            logger.info(
                f"  Frame {i}: inference #{num_inferences}, "
                f"actions {actions.shape}, "
                f"range [{actions.min():.4f}, {actions.max():.4f}], "
                f"server time {resp['inference_time']}s, "
                f"round-trip {dt:.2f}s"
            )

    # Reset
    post_json(f"{gateway_url}/reset", {"session_id": session_id})
    logger.info("Session reset. Done.")


def demo_batch_mode(gateway_url: str, prompt: str, use_zero: bool):
    """Send multiple frames at once via /predict_batch.

    This is more efficient when you already have multiple frames buffered.
    """
    logger.info("=== Batch mode ===")

    resp = post_json(f"{gateway_url}/start", {"prompt": prompt})
    session_id = resp["session_id"]
    logger.info(f"Session started: {session_id}")

    h, w = 180, 320

    if use_zero:
        make_frames = lambda n: np.zeros((n, h, w, 3), dtype=np.uint8)
    else:
        cameras = {}
        for cam, fname in CAMERA_FILES.items():
            cameras[cam] = load_video_frames(os.path.join(VIDEO_DIR, fname))

    # Step 0: send 1 frame
    logger.info("Step 0: sending 1 initial frame")
    if use_zero:
        ext0 = ext1 = wrist = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        ext0, ext1, wrist = cameras["exterior_0"][0], cameras["exterior_1"][0], cameras["wrist"][0]

    t0 = time.time()
    resp = post_json(f"{gateway_url}/predict_batch", {
        "session_id": session_id,
        "exterior_0": ext0.tolist(),
        "exterior_1": ext1.tolist(),
        "wrist": wrist.tolist(),
    })
    dt = time.time() - t0
    actions = np.array(resp["actions"])
    logger.info(f"  Actions: {actions.shape}, time: {resp['inference_time']}s (round-trip {dt:.2f}s)")

    # Subsequent steps: send 4 frames at a time
    for chunk in range(3):
        start = 1 + chunk * 4
        end = start + 4
        logger.info(f"Step {chunk + 1}: sending frames [{start}:{end}]")

        if use_zero:
            ext0 = ext1 = wrist = make_frames(4)
        else:
            ext0 = cameras["exterior_0"][start:end]
            ext1 = cameras["exterior_1"][start:end]
            wrist = cameras["wrist"][start:end]

        t0 = time.time()
        resp = post_json(f"{gateway_url}/predict_batch", {
            "session_id": session_id,
            "exterior_0": ext0.tolist(),
            "exterior_1": ext1.tolist(),
            "wrist": wrist.tolist(),
        })
        dt = time.time() - t0
        actions = np.array(resp["actions"])
        logger.info(f"  Actions: {actions.shape}, time: {resp['inference_time']}s (round-trip {dt:.2f}s)")

    post_json(f"{gateway_url}/reset", {"session_id": session_id})
    logger.info("Session reset. Done.")


def main():
    parser = argparse.ArgumentParser(description="DreamZero gateway client example")
    parser.add_argument("--gateway", default="http://localhost:8080", help="Gateway URL")
    parser.add_argument("--prompt", default="pick up the object", help="Task prompt")
    parser.add_argument("--use-zero-images", action="store_true", help="Use dummy zero frames")
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "both"],
        default="both",
        help="Demo mode (default: both)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.mode in ("single", "both"):
        demo_single_frame_mode(args.gateway, args.prompt, args.use_zero_images)

    if args.mode in ("batch", "both"):
        demo_batch_mode(args.gateway, args.prompt, args.use_zero_images)


if __name__ == "__main__":
    main()
