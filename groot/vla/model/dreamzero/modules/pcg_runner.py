"""Piecewise CUDA Graph (PCG) Runner for DreamZero DiT.

Eliminates CPU kernel launch overhead by capturing _forward_blocks as a
series of CUDA graphs, with NCCL all_reduce calls executed eagerly between
graph segments.

Usage:
    Set ENABLE_PCG=true environment variable to enable.
    PCG replaces torch.compile for TP inference.

How it works:
    1. On first inference call, runs one eager warmup pass.
    2. Re-runs the forward with CUDA graph capture.  At each all_reduce
       call, the capture is paused (capture_end), the all_reduce is executed
       eagerly, and a new capture is started (capture_begin).
    3. This produces N+1 graph segments for N all_reduce calls.
    4. On subsequent calls, the varying inputs (x, timestep, action,
       timestep_action) are copied into static buffers and the graphs are
       replayed with eager all_reduce between them.

For 32 DiT blocks with 3 all_reduce each = 96 AR points = 97 graph segments.
Each graph replay costs ~3-5us, so 97 replays add only ~0.3-0.5ms overhead.
"""

import torch
import torch.cuda
import torch.distributed as dist


class PCGRunner:
    """Drop-in replacement for model._forward_blocks with CUDA graph replay."""

    def __init__(self, model, tp_group, original_forward_blocks):
        self.model = model
        self.tp_group = tp_group
        self._original_forward = original_forward_blocks

        # Capture state
        self._captured = False
        self._graphs = []       # list of torch.cuda.CUDAGraph
        self._ar_buffers = []   # list of (tensor_ref, group) between graphs
        self._pool = None

        # Static input buffers (allocated during capture)
        self._static_x = None
        self._static_timestep = None
        self._static_action = None
        self._static_timestep_action = None
        # Fixed input buffers (don't change during denoising loop)
        self._static_freqs = None
        self._static_context = None
        self._static_clip_feature = None
        self._static_embodiment_id = None
        self._static_state = None
        self._static_kv_cache = None

        # Static output references (point into graph memory)
        self._static_output = None

        # Internal state for capture hooks
        self._mode = 'EAGER'  # 'EAGER' or 'CAPTURE'
        self._current_graph = None
        self._hooked_modules = []

        # Fixed scalar args (saved during capture)
        self._fixed_seq_len = None
        self._fixed_start_frame = None

        # Shape signature for re-capture detection
        self._shape_sig = None

    def _get_shape_sig(self, x, timestep, action, timestep_action, kv_cache):
        """Shape signature to detect when re-capture is needed."""
        sig = [x.shape]
        if timestep is not None:
            sig.append(timestep.shape)
        if action is not None:
            sig.append(action.shape)
        if timestep_action is not None:
            sig.append(timestep_action.shape)
        if kv_cache and len(kv_cache) > 0 and kv_cache[0] is not None:
            sig.append(kv_cache[0].shape)
        return tuple(sig)

    def _install_hooks(self):
        """Replace _all_reduce on all modules with a hook that breaks capture."""
        runner = self

        def hooked_all_reduce(tensor, group=None):
            if runner._mode == 'CAPTURE':
                # End current graph segment
                runner._current_graph.capture_end()
                runner._graphs.append(runner._current_graph)
                runner._ar_buffers.append((tensor, group))

                # Execute all_reduce eagerly (outside graph capture).
                # Use async_op + work.wait() so the NCCL stream fully
                # completes before we start the next graph segment.  Without
                # wait(), NCCL keeps an async reference to tensor (in pool
                # memory), and if _reset() frees the pool before NCCL
                # finishes, we get a SIGSEGV.
                work = dist.all_reduce(tensor, group=group, async_op=True)
                work.wait()

                # Start new graph segment
                runner._current_graph = torch.cuda.CUDAGraph()
                runner._current_graph.capture_begin(pool=runner._pool)
            else:
                # EAGER mode: normal all_reduce
                dist.all_reduce(tensor, group=group)

        self._hooked_modules = []
        for module in self.model.modules():
            # Check if the module's class defines _all_reduce (our hook point)
            if hasattr(module.__class__, '_all_reduce') and callable(
                getattr(module.__class__, '_all_reduce', None)
            ):
                # Replace at instance level (overrides class method)
                module._all_reduce = hooked_all_reduce
                self._hooked_modules.append(module)

    def _uninstall_hooks(self):
        """Remove instance-level _all_reduce overrides, restoring class methods."""
        for module in self._hooked_modules:
            if '_all_reduce' in module.__dict__:
                del module.__dict__['_all_reduce']
        self._hooked_modules = []

    def _warmup_and_capture(self, x, seq_len, freqs, timestep, context,
                            clip_feature, embodiment_id, action,
                            timestep_action, state, kv_cache,
                            current_start_frame):
        """Warmup + piecewise CUDA graph capture."""

        # Install hooks on all modules
        self._install_hooks()

        # --- Step 1: Warmup (eager) ---
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print("[PCG] Warmup: running eager forward pass...")
        self._mode = 'EAGER'
        with torch.no_grad():
            warmup_output = self._original_forward(
                x, seq_len, freqs, timestep, context, clip_feature,
                embodiment_id, action, timestep_action, state, kv_cache,
                current_start_frame,
            )

        # --- Step 2: Allocate static buffers ---
        # Varying inputs (change each denoising step)
        self._static_x = x.clone()
        self._static_timestep = timestep.clone() if timestep is not None else None
        self._static_action = action.clone() if action is not None else None
        self._static_timestep_action = (
            timestep_action.clone() if timestep_action is not None else None
        )
        # Fixed inputs (same across denoising steps within a chunk)
        self._static_freqs = freqs.clone()
        self._static_context = context.clone()
        self._static_clip_feature = (
            clip_feature.clone() if clip_feature is not None else None
        )
        self._static_embodiment_id = (
            embodiment_id.clone() if embodiment_id is not None else None
        )
        self._static_state = state.clone() if state is not None else None
        self._static_kv_cache = [
            kv.clone() if kv is not None else None for kv in kv_cache
        ]
        # Save fixed scalars
        self._fixed_seq_len = seq_len
        self._fixed_start_frame = current_start_frame

        # --- Step 3: Capture ---
        if rank == 0:
            print("[PCG] Capturing piecewise CUDA graphs...")
        self._mode = 'CAPTURE'
        self._graphs = []
        self._ar_buffers = []
        self._pool = torch.cuda.graph_pool_handle()

        capture_stream = torch.cuda.Stream()

        with torch.cuda.stream(capture_stream):
            # Wait for default stream work to finish
            capture_stream.wait_stream(torch.cuda.current_stream())

            # Start first graph segment
            self._current_graph = torch.cuda.CUDAGraph()
            self._current_graph.capture_begin(pool=self._pool)

            with torch.no_grad():
                output = self._original_forward(
                    self._static_x,
                    self._fixed_seq_len,
                    self._static_freqs,
                    self._static_timestep,
                    self._static_context,
                    self._static_clip_feature,
                    self._static_embodiment_id,
                    self._static_action,
                    self._static_timestep_action,
                    self._static_state,
                    self._static_kv_cache,
                    self._fixed_start_frame,
                )

            # End last graph segment
            self._current_graph.capture_end()
            self._graphs.append(self._current_graph)

        # Sync capture stream back to default
        torch.cuda.current_stream().wait_stream(capture_stream)

        self._current_graph = None
        self._mode = 'EAGER'

        # Save static output references
        self._static_output = output
        self._captured = True
        self._shape_sig = self._get_shape_sig(
            x, timestep, action, timestep_action, kv_cache
        )

        n_graphs = len(self._graphs)
        n_ar = len(self._ar_buffers)
        if rank == 0:
            print(
                f"[PCG] Captured {n_graphs} graph segments "
                f"with {n_ar} all_reduce points"
            )

        import sys
        print(f"[PCG dbg rank {rank}] A: before sync", flush=True)
        torch.cuda.synchronize()
        print(f"[PCG dbg rank {rank}] B: after sync", flush=True)
        if dist.is_initialized():
            dist.barrier()
        print(f"[PCG dbg rank {rank}] C: after barrier", flush=True)
        x_v, a_n, kvs = warmup_output
        print(f"[PCG dbg rank {rank}] D: warmup_output unpacked, x_v.shape={x_v.shape}", flush=True)
        xc = x_v.clone()
        print(f"[PCG dbg rank {rank}] E: x_video cloned", flush=True)
        ac = a_n.clone() if a_n is not None else None
        print(f"[PCG dbg rank {rank}] F: action cloned", flush=True)
        kvc = [kv.clone() if kv is not None else None for kv in kvs]
        print(f"[PCG dbg rank {rank}] G: kv_caches cloned ({len(kvc)} entries)", flush=True)
        return (xc, ac, kvc)

    def _replay(self):
        """Replay all captured graph segments with eager all_reduce between them."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[PCG dbg rank {rank}] _replay start, {len(self._graphs)} graphs", flush=True)
        for i, graph in enumerate(self._graphs):
            graph.replay()
            if i < len(self._ar_buffers):
                tensor, group = self._ar_buffers[i]
                print(f"[PCG dbg rank {rank}] replay AR {i}, tensor.shape={tensor.shape}", flush=True)
                # wait() ensures the all_reduce completes on the NCCL stream
                # before the next graph segment reads the tensor.
                work = dist.all_reduce(tensor, group=group, async_op=True)
                work.wait()
        print(f"[PCG dbg rank {rank}] _replay done", flush=True)

    def _copy_varying_inputs(self, x, timestep, action, timestep_action):
        """Copy new values into static buffers before replay."""
        self._static_x.copy_(x)
        if timestep is not None and self._static_timestep is not None:
            self._static_timestep.copy_(timestep)
        if action is not None and self._static_action is not None:
            self._static_action.copy_(action)
        if timestep_action is not None and self._static_timestep_action is not None:
            self._static_timestep_action.copy_(timestep_action)

    def _clone_output(self, output):
        """Clone output tensors from static graph memory.

        x_video and action_noise_pred live in the graph's memory pool and must
        be cloned before the pool can be reused.  updated_kv_caches also lives
        in the pool (created by torch.stack during capture), so it must be
        cloned too — otherwise callers would hold dangling references after a
        pool reset triggered by a shape change.
        """
        x_video, action_noise_pred, updated_kv_caches = output
        return (
            x_video.clone(),
            action_noise_pred.clone() if action_noise_pred is not None else None,
            [kv.clone() if kv is not None else None for kv in updated_kv_caches],
        )

    def _reset(self):
        """Reset capture state (for re-capture on shape change)."""
        # Flush all pending NCCL operations before releasing the graph pool.
        # dist.barrier() goes through NCCL and is serialized after all prior
        # all_reduce calls, ensuring NCCL's internal stream has fully drained
        # and released its references to pool-memory tensors.
        # torch.cuda.synchronize() then ensures the GPU-side CUDA ops are also
        # complete before we drop the graph/pool references.
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize()
        self._static_output = None   # release pool refs before clearing graphs
        self._graphs = []
        self._ar_buffers = []
        self._captured = False
        self._shape_sig = None
        self._pool = None

    def __call__(self, x, seq_len, freqs, timestep, context, clip_feature,
                 embodiment_id, action, timestep_action, state, kv_cache,
                 current_start_frame):
        """Drop-in replacement for model._forward_blocks."""

        shape_sig = self._get_shape_sig(
            x, timestep, action, timestep_action, kv_cache
        )

        if not self._captured or shape_sig != self._shape_sig:
            if self._captured:
                rank = dist.get_rank() if dist.is_initialized() else 0
                if rank == 0:
                    print("[PCG] Input shapes changed, re-capturing...")
                self._reset()
            return self._warmup_and_capture(
                x, seq_len, freqs, timestep, context, clip_feature,
                embodiment_id, action, timestep_action, state, kv_cache,
                current_start_frame,
            )

        # Fast path: copy varying inputs and replay
        self._copy_varying_inputs(x, timestep, action, timestep_action)
        self._replay()
        return self._clone_output(self._static_output)
