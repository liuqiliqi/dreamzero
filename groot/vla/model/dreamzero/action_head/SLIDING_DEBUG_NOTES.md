# Sliding vs Baseline 差异排查清单

> 目标：让 sliding 模式的行为等价于 baseline。目前 `SLIDE_EMPTY_CACHE=1 + SLIDE_RESET_LOW=1` 下，~8s（chunk 5-6）附近仍有卡顿。逐项消除 sliding 和 baseline 的差异，直到卡顿消失。

## 当前状态

**可以 work 的配置**：
- `USE_KVPRESS=0`（baseline，clean 版本）—— 完全正常

**仍有卡顿的配置**：
- `USE_KVPRESS=1 + KV_WINDOW_SIZE=2 + KV_SINK_SIZE=1` (re-rope 完整版本) — 严重卡顿，chunks 5+ 基本不动
- `USE_KVPRESS=1 + SLIDE_EMPTY_CACHE=1` (csf=6 on slide) — 仍卡顿
- `USE_KVPRESS=1 + SLIDE_EMPTY_CACHE=1 + SLIDE_RESET_LOW=1` (csf=2 on slide) — **当前版本**，~8s 仍卡顿，但据用户反馈比前面好一些

---

## 逐项差异（baseline vs 当前 sliding，`csf=2 + empty cache`）

在每个 chunk 进入 `lazy_joint_video_action` 时：

| # | 维度 | baseline | 当前 sliding (csf=2) | 影响程度假设 |
|---|---|---|---|---|
| 1 | **csf 初始值** | reset 到 0 | slide 设到 2 | 中 — 后续走 path 不同 |
| 2 | **走哪条 prefill path** | `if csf==0:` 初始 prefill 分支（line 1244） | `elif csf != 1:` recondition 分支（line 1273） | **大** |
| 3 | **Prefill 的 noisy_input 来源** | `new_image` from `encode_image`：1 latent ts，**单帧 real obs 的 VAE 编码** | `image[:, -npfb:]` from `vae.encode(videos[-1:].repeat(9))`：2 latent ts，**9 份相同当前帧的 VAE 编码末尾 2 帧** | **大** — 完全不同的 VAE 分布 |
| 4 | **Prefill 的 y** | `ys[:, :, 0:1]` (pos 0, msk=1, real) | `ys[:, :, 0:2]` (pos 0 real + pos 1 zero-pad) | 中 — 多了个 zero-pad 时步 |
| 5 | **Prefill 帧数 (写入 KV cache)** | 1 帧 at pos 0 | 2 帧 at pos 0, 1 | 小 |
| 6 | **Main diff 时的 csf** | 1 | 2 | 小 |
| 7 | **Main diff query 位置** | pos 1, 2 | pos 2, 3 | 小 |
| 8 | **Main diff 读的 cache pos** | pos 0 (1 帧) | pos 0, 1 (2 帧) | 小 |
| 9 | **`_create_kv_caches` / `_create_crossattn_caches` 调用** | 每 chunk 在 csf==0 path 里重新创建 | 只在首次 csf==0（chunk 0）创建，之后 slide 用 `SLIDE_EMPTY_CACHE` 手动清空（仅 self-attn，不动 crossattn） | 中 — crossattn_cache 在 sliding 里不重建 |
| 10 | **`clip_feas` / `self.ys` 来源** | 每 chunk encode_image 刷新（因为走 csf==0 path） | 每 chunk `_need_reencode=True` 触发 encode_image 刷新（我的 fix #1） | 应该等价 |
| 11 | **`if csf==1: cat image to output`** | 触发（csf=1） | 不触发（csf=2），靠我的 fix #2 的 `elif USE_KVPRESS==1 and csf!=0: cat image[:, :1]` | 已补 — 但 image 来源是 9-copy VAE，跟 baseline 的 new_image 不同 |
| 12 | **DUP_FIRST_FRAME 逻辑** | 不走（csf==0，不进入 `elif csf != 0:` 视频分支） | 走 `elif csf != 0:` 的单帧 `videos[:, :, -1:].repeat(9)` 分支，`DUP_FIRST_FRAME` 控制是否 `cat([first, videos])` 多加一帧 | 小 — 当前配置 DUP=1 |

---

## 排查建议顺序（假设影响大小排序）

**最大嫌疑**：#2 + #3 — prefill 的整个 path 不同，输入内容分布不同
- **排查动作**：sliding 模式强制走 csf==0 初始 prefill 分支。要求 slide 后 `csf=0`，走 `encode_image` 得 `new_image` 再做 init prefill
- 但 `csf=0` 会触发 `_create_kv_caches` 清 cache，基本等于 baseline
- 折中：加一个 flag `SLIDE_FORCE_INIT_PREFILL=1`，slide 后跑一次"假的初始 prefill"（csf=0 + `new_image` 作为 noisy_input + `ys[:, :, 0:1]` 作为 y + start_frame=0 + update_kv_cache=True），跑完再把 csf 设成 1 进入 main diff

**次要嫌疑**：#9 — crossattn_cache 在 sliding 里不重建
- 文本 K/V 理论上不受 frame RoPE 影响，重建后应该得到相同结果
- 但可以测：加 flag `SLIDE_RECREATE_CROSSATTN=1`，每次 slide 强制重建 crossattn_cache

**第三**：#11 — cat 的 image 来源
- Baseline cat 的是 `new_image`（单帧 clean VAE），sliding cat 的是 `image[:, :1]`（9-copy VAE 的第一帧）
- 这两个在 VAE 角度应该有差异（时间维度 9 和 1 编码不同）
- **排查动作**：sliding 模式下，cat 时也走一次 encode_image 拿 new_image，而不是用 9-copy 的 image[:, :1]

**小嫌疑**：#6/#7 — csf 差 1
- 已经从 6 降到 2，再降到 1 理论上也可以
- 但 csf=1 加 recondition 是 `ys[:, :, -1:0]` 空切片，会挂
- 需要改成 "csf=1 时跳过 recondition，直接去做 initial prefill"

---

## 已经排除的假设

- ❌ **bf16 量化累积**：SLIDE_EMPTY 清空 cache 后仍卡 → 不是 re-rope bf16 噪声
- ❌ **KV content staleness**：SLIDE_EMPTY 已清 cache 内容 → 不是旧内容污染
- ❌ **csf 高位 "end-of-task" heuristic**：SLIDE_RESET_LOW 把 csf 降到 2 仍卡 → 不是单纯 csf 数值问题，可能是 csf + 其他因素共同作用
- ❌ **文本注入路径被 RoPE 扭曲**：`_slide_kv_window` 只动 self-attention 两个 cache，crossattn_cache 未触碰，文本 embedding 不用 frame RoPE

---

## 固定不动的配置（验证时保持）

```bash
USE_KVPRESS=1 KV_WINDOW_SIZE=2 KV_SINK_SIZE=1
SLIDE_EMPTY_CACHE=1 SLIDE_RESET_LOW=1
DUP_FIRST_FRAME=1
BIG_MUG=1
NUM_DIT_STEPS=16 ATTENTION_BACKEND=FA2
gripper 阈值 = 0.2
```

在 `ultimate-law` 节点跑，1 ep 验证每次改动。

---

## 当前下一步

用户反馈 ~8s（chunk 5-6）仍卡顿。按照"最大嫌疑"排查顺序：加 `SLIDE_FORCE_INIT_PREFILL=1`，让 slide 后走 baseline 的初始 prefill 分支（用 new_image 而不是 9-copy VAE）。
