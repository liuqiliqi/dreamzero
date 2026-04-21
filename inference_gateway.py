#!/usr/bin/env python3
"""Inference gateway: simple API for external devices to use DreamZero.

Architecture:
    External Device  <--HTTP/JSON-->  Gateway (this)  <--WebSocket/msgpack-->  DreamZero Server

The gateway handles:
    - Frame accumulation (external device sends 1 frame at a time, gateway buffers 4)
    - Image resizing to model resolution (180x320)
    - Session management
    - Protocol translation

Usage:
    # 1. Make sure DreamZero server is running:
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
        socket_test_optimized_AR.py --port 5000 --enable-dit-cache \
        --model-path /fact_data/qiliu/dreamzero_weights/DreamZero-DROID

    # 2. Start the gateway:
    python inference_gateway.py --dreamzero-port 5000 --port 8080

    # 3. External devices connect to the gateway (see gateway_client_example.py)
"""

import argparse
import json
import logging
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np

from eval_utils.policy_client import WebsocketClientPolicy

logger = logging.getLogger(__name__)

# DreamZero expects 180x320 (H x W)
MODEL_H, MODEL_W = 180, 320

# Each inference produces 24 action steps
ACTION_HORIZON = 24

# After the initial frame, we accumulate this many frames before each inference
FRAMES_PER_CHUNK = 4


class DreamZeroSession:
    """Manages one inference session with the DreamZero server."""

    def __init__(self, client: WebsocketClientPolicy, prompt: str):
        self.client = client
        self.prompt = prompt
        self.session_id = str(uuid.uuid4())
        self.step = 0  # 0 = initial, 1+ = subsequent chunks
        self.frame_buffer: dict[str, list[np.ndarray]] = {
            "exterior_0": [],
            "exterior_1": [],
            "wrist": [],
        }
        self.lock = threading.Lock()
        self.last_actions: np.ndarray | None = None

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Resize image to model resolution if needed."""
        h, w = img.shape[:2]
        if h != MODEL_H or w != MODEL_W:
            img = cv2.resize(img, (MODEL_W, MODEL_H))
        return img

    def add_frame(self, exterior_0: np.ndarray, exterior_1: np.ndarray, wrist: np.ndarray):
        """Add one frame from each camera to the buffer."""
        with self.lock:
            self.frame_buffer["exterior_0"].append(self._resize(exterior_0))
            self.frame_buffer["exterior_1"].append(self._resize(exterior_1))
            self.frame_buffer["wrist"].append(self._resize(wrist))

    def ready_to_infer(self) -> bool:
        """Check if we have enough frames for the next inference."""
        with self.lock:
            needed = 1 if self.step == 0 else FRAMES_PER_CHUNK
            return len(self.frame_buffer["exterior_0"]) >= needed

    def infer(self) -> np.ndarray:
        """Run inference with buffered frames. Returns (24, 8) action array."""
        with self.lock:
            if self.step == 0:
                # Initial: take 1 frame per camera → (H, W, 3)
                frames = {k: v.pop(0) for k, v in self.frame_buffer.items()}
            else:
                # Subsequent: take 4 frames per camera → (4, H, W, 3)
                frames = {}
                for k, v in self.frame_buffer.items():
                    frames[k] = np.stack(v[:FRAMES_PER_CHUNK], axis=0)
                    del v[:FRAMES_PER_CHUNK]

        # Build observation dict in roboarena format
        obs = {
            "observation/exterior_image_0_left": frames["exterior_0"],
            "observation/exterior_image_1_left": frames["exterior_1"],
            "observation/wrist_image_left": frames["wrist"],
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/cartesian_position": np.zeros(6, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
            "prompt": self.prompt,
            "session_id": self.session_id,
        }

        actions = self.client.infer(obs)
        self.step += 1
        self.last_actions = actions
        return actions

    def reset(self):
        """Reset the session (triggers video save on server)."""
        self.client.reset({})
        self.step = 0
        self.frame_buffer = {k: [] for k in self.frame_buffer}
        self.last_actions = None


class GatewayState:
    """Global state for the gateway server."""

    def __init__(self, dreamzero_host: str, dreamzero_port: int):
        self.dreamzero_host = dreamzero_host
        self.dreamzero_port = dreamzero_port
        self.sessions: dict[str, DreamZeroSession] = {}
        self.lock = threading.Lock()

    def create_session(self, prompt: str) -> DreamZeroSession:
        client = WebsocketClientPolicy(host=self.dreamzero_host, port=self.dreamzero_port)
        session = DreamZeroSession(client, prompt)
        with self.lock:
            self.sessions[session.session_id] = session
        logger.info(f"Created session {session.session_id} with prompt: {prompt}")
        return session

    def get_session(self, session_id: str) -> DreamZeroSession | None:
        with self.lock:
            return self.sessions.get(session_id)

    def remove_session(self, session_id: str):
        with self.lock:
            self.sessions.pop(session_id, None)


# Global state (set in main)
_state: GatewayState = None


class GatewayHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the gateway.

    Endpoints:
        POST /start
            Body: {"prompt": "pick up the cup"}
            Response: {"session_id": "xxx"}

        POST /predict
            Body: {"session_id": "xxx",
                   "exterior_0": [[[r,g,b], ...], ...],   # H x W x 3 uint8 list
                   "exterior_1": [[[r,g,b], ...], ...],
                   "wrist":      [[[r,g,b], ...], ...],
                   "state": {"joint_position": [...], "gripper_position": [...]}  # optional
                  }
            Response: {"actions": [[...], ...], "shape": [24, 8]}

        POST /predict_batch
            Body: {"session_id": "xxx",
                   "exterior_0": [[frame0], [frame1], ...],  # N x H x W x 3
                   "exterior_1": [[frame0], [frame1], ...],
                   "wrist":      [[frame0], [frame1], ...]}
            Response: {"actions": [[...], ...], "shape": [24, 8]}

        POST /reset
            Body: {"session_id": "xxx"}
            Response: {"status": "ok"}

        GET /status
            Response: {"sessions": [...], "model_resolution": [180, 320]}
    """

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw)

    def do_GET(self):
        if self.path == "/status":
            with _state.lock:
                session_ids = list(_state.sessions.keys())
            self._send_json({
                "sessions": session_ids,
                "model_resolution": [MODEL_H, MODEL_W],
                "action_horizon": ACTION_HORIZON,
                "frames_per_chunk": FRAMES_PER_CHUNK,
            })
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            if self.path == "/start":
                self._handle_start()
            elif self.path == "/predict":
                self._handle_predict()
            elif self.path == "/predict_batch":
                self._handle_predict_batch()
            elif self.path == "/reset":
                self._handle_reset()
            else:
                self._send_json({"error": "not found"}, 404)
        except Exception as e:
            logger.exception("Error handling request")
            self._send_json({"error": str(e)}, 500)

    def _handle_start(self):
        data = self._read_json()
        prompt = data.get("prompt", "pick up the object")
        session = _state.create_session(prompt)
        self._send_json({"session_id": session.session_id})

    def _handle_predict(self):
        """Accept 1 frame per camera. Buffer until ready, then infer."""
        data = self._read_json()
        session = _state.get_session(data["session_id"])
        if session is None:
            self._send_json({"error": "session not found"}, 404)
            return

        # Parse images: expect H x W x 3 uint8 nested lists
        ext0 = np.array(data["exterior_0"], dtype=np.uint8)
        ext1 = np.array(data["exterior_1"], dtype=np.uint8)
        wrist = np.array(data["wrist"], dtype=np.uint8)
        session.add_frame(ext0, ext1, wrist)

        if not session.ready_to_infer():
            buffered = len(session.frame_buffer["exterior_0"])
            needed = 1 if session.step == 0 else FRAMES_PER_CHUNK
            self._send_json({
                "status": "buffering",
                "buffered": buffered,
                "needed": needed,
                "message": f"Need {needed - buffered} more frame(s) before inference",
            })
            return

        t0 = time.time()
        actions = session.infer()
        dt = time.time() - t0

        self._send_json({
            "status": "ok",
            "actions": actions.tolist(),
            "shape": list(actions.shape),
            "inference_time": round(dt, 3),
            "step": session.step,
        })

    def _handle_predict_batch(self):
        """Accept N frames per camera at once, buffer all, then infer if ready."""
        data = self._read_json()
        session = _state.get_session(data["session_id"])
        if session is None:
            self._send_json({"error": "session not found"}, 404)
            return

        ext0_batch = np.array(data["exterior_0"], dtype=np.uint8)
        ext1_batch = np.array(data["exterior_1"], dtype=np.uint8)
        wrist_batch = np.array(data["wrist"], dtype=np.uint8)

        # If 3D (H,W,3), treat as single frame
        if ext0_batch.ndim == 3:
            ext0_batch = ext0_batch[np.newaxis]
            ext1_batch = ext1_batch[np.newaxis]
            wrist_batch = wrist_batch[np.newaxis]

        for i in range(ext0_batch.shape[0]):
            session.add_frame(ext0_batch[i], ext1_batch[i], wrist_batch[i])

        if not session.ready_to_infer():
            buffered = len(session.frame_buffer["exterior_0"])
            needed = 1 if session.step == 0 else FRAMES_PER_CHUNK
            self._send_json({
                "status": "buffering",
                "buffered": buffered,
                "needed": needed,
            })
            return

        t0 = time.time()
        actions = session.infer()
        dt = time.time() - t0

        self._send_json({
            "status": "ok",
            "actions": actions.tolist(),
            "shape": list(actions.shape),
            "inference_time": round(dt, 3),
            "step": session.step,
        })

    def _handle_reset(self):
        data = self._read_json()
        session = _state.get_session(data["session_id"])
        if session is None:
            self._send_json({"error": "session not found"}, 404)
            return
        session.reset()
        _state.remove_session(data["session_id"])
        self._send_json({"status": "ok"})

    def log_message(self, format, *args):
        logger.info(format, *args)


def main():
    global _state

    parser = argparse.ArgumentParser(description="DreamZero inference gateway")
    parser.add_argument("--port", type=int, default=8080, help="Gateway port (default: 8080)")
    parser.add_argument("--dreamzero-host", default="localhost", help="DreamZero server host")
    parser.add_argument("--dreamzero-port", type=int, default=5000, help="DreamZero server port")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    _state = GatewayState(args.dreamzero_host, args.dreamzero_port)

    server = HTTPServer(("0.0.0.0", args.port), GatewayHandler)
    logger.info(f"Gateway listening on port {args.port}")
    logger.info(f"DreamZero backend: {args.dreamzero_host}:{args.dreamzero_port}")
    logger.info(f"Endpoints: POST /start, /predict, /predict_batch, /reset | GET /status")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
