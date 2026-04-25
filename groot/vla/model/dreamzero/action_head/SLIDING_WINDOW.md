# KVpress 滑窗 + RoPE 位移重编码（W=2 版本）

> 2026-04-24 更新：补充首帧失真根因 + 两个修复，以及 slide 时间点对应 chunk 编号。

## 动机

`USE_KVPRESS=1` 在 KV cache 满了之后**保留最近几帧的 K/V**，避免像 reset 那样丢光历史。原实现（`self.current_start_frame = keep_frames`）只把 `csf` 计数器拉回，**没有对 cache 里已经"烤进去"的 RoPE 位置做任何处理**，导致：

- 保留下来的 K 位置编号仍然是它们被写入时的老值（比如 `[5, 6, 7, 8]`）
- 新 prefill 写的位置按新 `csf` 计算（比如 `[3, 4]`），**和老位置非单调、有冲突**
- 注意力分数变成垃圾，机械臂动作被抑制 / 抓取率下降

---

## 参数（DROID 场景）

| | |
|---|---|
| `num_frame_per_block (npfb)` | 2 |
| `max_chunk_size` | 4 |
| `local_attn_size` | 2×4 + 1 = **9** |
| `KV_WINDOW_SIZE (W)` | **2**（推荐） |
| `KV_SINK_SIZE` | 1 |
| `kept_total` | W × npfb = **4** 帧 |
| `kept_recent` | kept_total - sink = 3 |

---

## Cache 布局（slide 后稳态）

```
 pos     0    1    2    3    |   4  5  (next chunk prefill 写这里)
        [S]  [R0] [R1] [R2]  |
        sink   recent
```

- `S` = 第 0 帧（初始观察，"task anchor"）—— RoPE 位置保持原样
- `R0..R2` = 最后 3 帧被**重定位**到 pos 1/2/3（乘一个共享 correction）

---

## RoPE 位移重编码（核心数学）

RoPE 在复数空间：`K_at_pos_p = K_raw ⊙ freqs[p]`，其中 `freqs[p] = exp(i · p · θ_d)`。

要把烤在 K 里的老位置 `p_old` 变成新位置 `p_new`：
```
K_new = K_old × (freqs[p_new] / freqs[p_old]) 
      = K_old × freqs[p_new - p_old]
      = K_old × freqs[shift]
```

所有 recent 帧共享同一个 `shift = frames_in_cache - kept_recent - sink_size`，稳态 W=2 时 `shift = 8 - 3 - 1 = 4`。只需乘一个**共享 correction** `conj(freqs[shift])`。

**关键约束**：
- 只对 **K** 做 re-rope（V 没 RoPE）
- 只对 **head_dim 前 `d0_complex` 维**（frame 轴的 RoPE 段，h/w 轴不碰）
- 用 **float64 精度**（训练时 `rope_params_polar` 的精度），变换后转回原 dtype

---

## Slide 时间点（DROID, 15 Hz, open_loop_horizon=24）

每 chunk = 24 sim steps / 15 Hz = **1.6 s**。450 steps/ep = 18.75 chunks ≈ 19/ep。

csf 演化（chunk 0 包括初始 prefill +1 + 主扩散 +2 共 +3）：

| chunk | entry csf | 动作 | exit csf | 时间 (s) |
|---|---|---|---|---|
| 0 | 0 | prefill+diff | 3 | 0.0–1.6 |
| 1 | 3 | diff | 5 | 1.6–3.2 |
| 2 | 5 | diff | 7 | 3.2–4.8 |
| 3 | 7 | diff | 9 | 4.8–6.4 |
| **4** | **9** | **slide→csf=6, diff** | **8** | **6.4–8.0** |
| 5 | 8 | diff | 10 | 8.0–9.6 |
| 6 | 10 | slide→6, diff | 8 | 9.6–11.2 |
| 7 | 8 | diff | 10 | 11.2–12.8 |
| 8 | 10 | slide | 8 | 12.8–14.4 |
| … | … | slide 每隔一个 chunk | | |

**第一次 slide 发生在 chunk 4**（6.4–8.0s），对应真实视频 **8s 附近的"夹爪莫名松开"行为**——slide 扰动 KV cache，action head 跟 video head 共用同一个 cache，action 预测可能因此翻转（例如 gripper open/close 的 sigmoid 跨过 0.5）。

---

## In-distribution 约束（W=2 的关键优势）

训练见过的 RoPE 位置范围：`[0, local_attn_size-1] = [0, 8]`。

main-loop query 最大位置 `= kept_total + 2·npfb - 1`：
- W=2: `4 + 4 - 1 = 7 ≤ 8` ✓ 
- W=3: `6 + 4 - 1 = 9` → OOD
- W=4: `8 + 4 - 1 = 11` → 严重 OOD → **上次跑出的"犹豫 / 抓取退化"就是这个**

所以 **W=2 是严格 in-distribution 的最大值**，推荐配置。

---

## 2026-04-24 修复记录（首帧失真 + 后续帧不一致）

用户观察：sliding 模式下 pred_video 每个 chunk (≥1) 的**首帧失真**，baseline 没有。

### 两个根因

**根因 1 — 视觉条件化 stale（影响后续帧）**：
- baseline 每 chunk reset csf=0 → 走 `encode_image` 路径 → `self.clip_feas` / `self.ys` 每次从当前观察刷新
- sliding 保留 csf，不再走 csf==0 分支 → `clip_feas` / `ys` 一直是 chunk 0 的陈旧值
- 后续 chunk 用 chunk 0 的视觉条件预测当前帧 → 颜色/内容偏差

**根因 2 — pred_video 首帧没 prepend 真实观察（核心根因）**：
- `if self.current_start_frame == 1: output = torch.cat([image, output], dim=1)` — 把 VAE 编码的当前观察 latent 塞到 video_pred 最前
- baseline csf 每次都是 1 → **每个 chunk 的 pred 首帧都是真实观察** → 干净
- sliding csf 不再是 1 → cat 被跳过 → pred 首帧变成 diffusion 产物 → 失真

### 修复（都在 `wan_flow_matching_action_tf.py`）

```python
# Fix 1: sliding 每个 csf≠0 chunk 都强制 re-encode clip/ys
if os.environ.get("USE_KVPRESS", "0") == "1" and self.current_start_frame != 0:
    self._need_reencode = True

# Fix 2: sliding 也 prepend 当前观察 latent 到 output
if self.current_start_frame == 1:
    output = torch.cat([image, output], dim=1)
elif os.environ.get("USE_KVPRESS", "0") == "1" and self.current_start_frame != 0:
    output = torch.cat([image[:, :1], output], dim=1)
```

---

## ip=2 并行下的坑

两个 rank 只有一个持有非空 cache，另一个 `seq_len=0`。两点要注意：

1. **跳过空 cache 的 re-rope math**（`if cache.shape[2] == 0: continue`），否则 `reshape(..., -1, 2)` 遇到 0 个元素会 raise
2. **`frames_in_cache` 必须从 populated 那个 cache 算**。如果两个 rank 从不同 cache 算出不同 `frames_in_cache`，各自的 `self.current_start_frame` 会分叉，main-loop 各自按不同 csf 计算 RoPE，CFG 合并 `uncond + 5·(cond-uncond)` 后就变成剧烈跳变的视频。表现：pred 视频 intra-chunk 亮度方差从 baseline 的 ~2 飙到 ~10

代码用 `for candidate in (self.kv_cache1, self.kv_cache_neg): if candidate[0].shape[2] > 0: sample = candidate[0]; break` 动态找非空 cache 规避。

---

## 其他注意事项

- **CrossAttn cache 不管**（跨注意力走文本 key/value，跟 frame 位置无关）
- **语言切换 / 首次 call** 仍走原来的 reset 分支（`csf=0`）
- **slide 频率**：W=2 下每隔一个 chunk 触发一次（csf 8→10→slide→6→8→10→slide）；W=4 每个 chunk 都触发，bf16 量化累积更严重

---

## 快速验证

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_DIT_STEPS=16 USE_KVPRESS=1 KV_WINDOW_SIZE=2 KV_SINK_SIZE=1 \
BIG_MUG=1 \
bash /fact_home/qiliu/worldmodel/scripts/run_sliding_spaced_w2.sh 2 1
```

服务端日志每隔一个 chunk 应该看到：
```
[KV slide+rerope+sink] shift=4, sink=1, recent=3, csf=6
[KV slide] re-encoded CLIP/ys from current observation
```

对比指标（scene=2，"put the can in the mug"）：
- baseline（`USE_KVPRESS=0`）：`joint_path_len ≈ 9.81`，`gripper_closed ≈ 40%`
- 原版 bug（`csf = keep_frames`，无 rerope）：`6.30` / `12%`
- rerope + **W=2**（严格 in-dist，带 2026-04-24 修复）：✅ 首帧干净 + 性能比 baseline 提升
- rerope + `W=4`（位置溢出到 10, 11）：`10.08` / `6%` — 动是动但抓取退化
