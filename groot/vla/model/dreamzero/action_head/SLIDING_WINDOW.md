# KVpress sliding-window + RoPE 位移重编码

## 动机

`USE_KVPRESS=1` 想在 KV cache 满了之后**保留最近几帧的 K/V**，避免像 reset 那样丢光历史。但原实现（`self.current_start_frame = keep_frames`）只把 `csf` 计数器拉回，**没有对 cache 里已经"烤进去"的 RoPE 位置做任何处理**，导致：

- 保留下来的 K 位置编号仍然是它们被写入时的老值（比如 `[5, 6, 7, 8]`）。
- 新 prefill 写的位置按新 `csf` 计算（比如 `[3, 4]`），**和老位置非单调、有冲突**。
- 注意力分数变成垃圾，机械臂动作被抑制 / 抓取率下降。

## 实现

**入口**：`_slide_kv_window(window_size)`（`wan_flow_matching_action_tf.py`）。

**核心思路 — RoPE 位置在复数空间是可平移的**：
`rope_params_polar` 返回的 `freqs[0][p, d] = exp(i · p · θ_d)`，所以
```
K_at_new_pos = K_at_old_pos · freqs[0][new_pos] · conj(freqs[0][old_pos])
             = K_at_old_pos · freqs[0][new_pos − old_pos]
```
所有保留的帧平移同一个量 Δ，只需要乘一个**共享的 correction** `conj(freqs[0][shift])`。

**关键步骤**：
1. 从 cache 里保留最后 `kept_frames` 帧（`= min(window_size × npfb, cache 里的帧数)`）。
2. 计算 `shift = csf − npfb − kept_frames`（老位置 `[csf−npfb−kept, csf−npfb−1]` 平移到 `[0, kept−1]`）。
3. 对每层 KV cache 里的 **K**（不是 V）、并且**只**乘到 head_dim 复数前 `d0_complex` 个维度（frame 轴的 RoPE 段；h/w 轴不碰）。
4. 设 `self.current_start_frame = kept_frames + npfb`，下一个 prefill 在 `[kept_frames, kept_frames+npfb−1]` 连续单调地写入。
5. Cache 位置始终是 `[0, 1, ..., kept_frames + npfb − 1]`（prefill 后）或 `[0, 1, ..., kept_frames − 1]`（slide 后）。

**Slide 触发点调整**（`lazy_joint_video_action`）：
```python
slide_threshold = local_attn_size - npfb   # kvpress 模式
                = local_attn_size           # reset 模式
```
这样 kvpress 会在 main loop query 达到 OOD 位置**之前**就触发 slide，每个 chunk 都 slide 一次。所有被用到的 RoPE 位置（cache 里的、新 prefill 的、main loop query 的）都 ≤ `local_attn_size − 1 = 8`，**严格在训练见过的分布内**。

## 推荐参数

模型 config: `num_frame_per_block = 2`, `max_chunk_size = 4` → `local_attn_size = 9`。

为了保证 main loop 最大 query 位置 `= kept_frames + 2·npfb − 1 ≤ local_attn_size − 1 = 8`：
```
kept_frames ≤ local_attn_size − 2·npfb = 5
kept_frames = KV_WINDOW_SIZE × npfb     # 实现里的公式
```
所以 `KV_WINDOW_SIZE ≤ 2`（对应 `kept_frames ≤ 4`）全部 in-distribution。**推荐 `KV_WINDOW_SIZE=2`**：
- Slide 后 cache 在位置 `[0, 1, 2, 3]`（正好 4 帧）。
- Prefill 写 `[4, 5]`，cache → `[0..5]`（6 帧）。
- Main loop 查 `[6, 7]`。
- 所有位置都 ≤ 8。

`KV_WINDOW_SIZE=1` 窗口更紧但历史 cache 只留 2 帧，上下文短。
`KV_WINDOW_SIZE ≥ 3` 会让 main query 跑到位置 9+ (OOD)，性能会退化。

## 其他注意事项

- **`ip_size = 2` 并行下**，每个 rank 只填 `kv_cache1` **或** `kv_cache_neg` 的其中一个，另一个保持 `seq_len=0`。有两个注意点：
  - 对空 cache 要跳过 re-rope math（`if cache.shape[2] == 0: continue`），否则 `reshape(..., -1, 2)` 遇到 0 个元素会 raise。
  - **更隐蔽的坑**：`frames_in_cache`（决定 `kept_frames` / `shift` / 新 `csf`）必须从**populated 的**那个 cache 算，不能写死 `self.kv_cache1[0]`。如果两个 rank 从不同 cache 算出不同的 `frames_in_cache`，它们各自的 `self.current_start_frame` 就会分叉，main-loop 各自按不同 csf 计算 RoPE，CFG 合并 `uncond + 5*(cond-uncond)` 后就变成剧烈跳变的视频。表现：pred 视频 intra-chunk 亮度方差从 baseline 的 ~2 飙到 ~10。代码里用一个 `for candidate in (self.kv_cache1, self.kv_cache_neg): if ... candidate[0].shape[2] > 0: sample = candidate[0]; break` 动态找非空 cache。
- **复数变换用 float64** 保精度，和训练时 `rope_params_polar` 精度一致；变换后再转回 K 的原 dtype（通常 bf16）。
- **只对 K 做 re-rope**，V 没 RoPE，不需要动。
- **CrossAttn cache 不管**，因为跨注意力走文本 key/value，跟 frame 位置无关。
- **语言切换 / 首次 call** 仍走原来的 reset 分支（`csf=0`）。

## 快速验证

跑法：
```bash
KV_WINDOW_SIZE=2 USE_KVPRESS=1 bash /tmp/run_direct.sh 2 10 1
```

服务端日志里每个 chunk 应该能看到：
```
[KV slide+rerope] shift=N, kept_frames=4, seq_len=3520, csf=6
```
或首次的 no-shift 版本。`csf=6` 是对的（`4 + 2 = 6`）。

对比指标（scene=2, 1 ep, 任务 "put the can in the mug"）：
- baseline（`USE_KVPRESS=0`）：`joint_path_len ≈ 9.81`，`gripper_closed ≈ 40%`。
- 原版 bug（`csf = keep_frames`）：`6.30` / `12%`。
- rerope + `W=4`（位置溢出到 10, 11）：`10.08` / `6%` — 动是动但抓取退化。
- rerope + `W=2`（严格 in-dist）：**待测**，理论上最接近 baseline。
