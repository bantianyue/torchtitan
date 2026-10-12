# qlora_lab — 纯 CPU 上微调 Qwen3.5-0.8B

在本机（**i7-1165G7，4 核 / 8 线程，16 GB，无 GPU**）对 Qwen3.5-0.8B 做 LoRA 监督微调，
并测量它到底能跑多快、吃多少内存。

## 为什么独立成目录

`torchtitan/` 是**预训练**框架：模型定义、张量分片、优化器全是自带的，没有 HF 权重加载、
没有 PEFT/LoRA、没有 4bit。所以这部分工作无法复用它的任何代码。
本目录与 `cpu_lab/` 平级，**不修改 `torchtitan/` 下的任何文件**，
并有自己的 `.venv`（因此也不会污染 `cpu_lab` 依赖的那个 venv）。

## 为什么不用 4bit 量化

4bit（QLoRA）的唯一目的是**省显存**，让大模型能进单卡。在本机：

- 没有 GPU，bitsandbytes 只能走 **CPU 回退内核**：每次矩阵乘前先反量化成 bf16，
  比原生 fp32 多一遍 dequant，**更慢**；
- 0.8B 主干用 fp32 也只有 ~3.4 GB，16 GB 内存本来就装得下。

所以 4bit 在这里省不出有意义的东西。真正省内存的是 **LoRA 本身**：
主干冻结 ⇒ 不产生梯度、不占优化器状态，只有适配器需要训练。
正式结论与实测数字见下方「实测结果」。

## 环境

| 项 | 值 |
|---|---|
| venv | `qlora_lab/.venv`（CPython 3.11，uv 创建） |
| torch | 2.14.1+cpu |
| transformers | 5.19.0（Qwen3.5 需要 **≥ 5.2.0**） |
| peft / accelerate / datasets | 0.21.2 / 1.15.0 / 5.1.0 |
| HF 缓存 | `qlora_lab/hf`（落在 D 盘；C 盘只剩 ~13 GiB） |

一键准备（幂等，可重复执行）：

```bash
bash qlora_lab/setup_env.sh 2>&1 | tee qlora_lab/setup.log
```

## 落盘位置（全部在 D 盘）

本机 C 盘只剩约 12 GiB，所以**所有产物都固定落到 D 盘**，脚本里已写死：

| 内容 | 位置 |
|---|---|
| venv | `qlora_lab/.venv` |
| uv 包缓存 | `qlora_lab/.uvcache`（`UV_CACHE_DIR` 指定；uv 默认会写到 `%LOCALAPPDATA%\uv\cache`） |
| HF / datasets 缓存 | `qlora_lab/hf`（`HF_HOME` 指定） |
| 模型权重 | `qlora_lab/model`（扁平目录，不依赖 HF 缓存的符号链接布局） |
| 运行产物 | `qlora_lab/runs/<tag>`、`qlora_lab/reports` |

**注意**：`uv` 与 `huggingface_hub` 默认都会往 C 盘写（前者写包缓存，后者写
`~/.cache/huggingface`）。所以任何新脚本都必须先设 `UV_CACHE_DIR` 与 `HF_HOME`，
否则会悄悄污染 C 盘。

## 下载权重

直连 `huggingface.co` 在本机取不到数据（curl 返回 http 000），且 hf-xet 后端会卡死。
权重又是**单个 1.747 GB 文件**，`snapshot_download` 的并行是「按文件并行」，
对单文件无效。可用的路径：

```bash
bash qlora_lab/fetch_model_mirror.sh     # 镜像 + 8 连接分段并行 + sha256 校验
```

它从 `hf-mirror.com` 取（实测 http 206、支持 `Accept-Ranges: bytes`），把大文件切成
8 段并发下载再拼接，最后用 sha256 与 HF 的 blob etag 比对。
实测聚合速率约 **1.9 MB/s**（本地链路本身约 2 MB/s，是瓶颈；ModelScope 更慢），
整个 1.7 GB 约 17 分钟。

若只想用标准下载器（含重试）：

```bash
HF_HUB_DISABLE_XET=1 qlora_lab/.venv/Scripts/python.exe qlora_lab/fetch_model.py
```

## 用法

```bash
PY=qlora_lab/.venv/Scripts/python.exe

# 1) 数据管线自检（不需要权重，验证 chat template 前缀对齐与 loss mask）
$PY qlora_lab/check_data.py

# 2) 结构探查（不分配权重内存）
$PY qlora_lab/inspect_model.py

# 3) dry-run：一次真实的前向+反向，报告耗时与峰值内存
$PY qlora_lab/train_lora.py --dry-run --seq-len 256 --out-dir qlora_lab/runs/dryrun

# 4) 短 demo
$PY qlora_lab/train_lora.py \
  --dataset yahma/alpaca-cleaned --n-samples 512 \
  --seq-len 256 --steps 20 --batch-size 1 --grad-accum 1 \
  --lora-r 8 --lora-alpha 16 --dtype float32 --threads 4 \
  --out-dir qlora_lab/runs/alpaca_20step --save-adapter

# 5) 出报告
$PY qlora_lab/report.py --run-dir qlora_lab/runs/alpaca_20step
```

## 模型结构（实测探查结果）

`Qwen3_5ForConditionalGeneration`，`model_type=qwen3_5`，配置是 `text_config` + `vision_config` 双塔。

| 部分 | 参数量 |
|---|---|
| 语言主干 | 752.4 M |
| 视觉塔 `model.visual` | 100.6 M |
| lm_head（与 embedding **共享权重**，tie_word_embeddings=True） | 254.3 M（不额外计入） |
| 合计（去重后） | ≈ 853 M |

文本侧：`hidden 1024`、`24 层`、`8 heads`（KV 2）、`head_dim 256`、`vocab 248,320`、
`max_position 262,144`。

**24 层不是同一种层**，`layer_types` 按 `线性注意力 ×3 + 全注意力 ×1` 循环（18 + 6）：

- 线性注意力层（Gated DeltaNet）：`linear_attn.{in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj}`
- 全注意力层：`self_attn.{q_proj, k_proj, v_proj, o_proj}`
- 两层都有：`mlp.{gate_proj, up_proj, down_proj}`

LoRA 目标层因此是：

```
q_proj,k_proj,v_proj,o_proj,          # 6 层全注意力
in_proj_qkv,in_proj_z,out_proj,       # 18 层 GDN
gate_proj,up_proj,down_proj           # 全部 24 层
```

`in_proj_a` / `in_proj_b` 输出仅 16 维（门控标量），**排除**——给它们加 rank=8 的适配器没有意义。

## CPU 上的坑（都是实测踩出来的）

1. **必须 `attn_implementation="eager"`。** CPU 上的 SDPA 会派发到
   `_scaled_dot_product_flash_attention_for_cpu`，这个后端**没有反向 kernel**，
   训练时直接崩。脚本已默认强制 eager。
2. **必须 `enable_input_require_grads()`。** LoRA 适配层位于冻结的 embedding 下游；
   不开这个，计算图里没有需要梯度的叶子节点，反向拿不到梯度。
3. **hf-xet 在本机会卡死。** 用 `snapshot_download` 默认走 Xet 后端时，blob 停在
   536 MB 后几分钟零增长、进程还活着。解法：`HF_HUB_DISABLE_XET=1` 走经典 HTTP 下载
   （注意此时**不会复用** Xet 留下的 `.incomplete`，会重下）。
4. **视觉塔是纯文本场景的死重**（100.6 M 参数 ≈ 400 MB fp32）。不喂图时它不会被
   执行，所以只占内存不占算力。
5. **数据集切片的长度分布要看清**：`yahma/alpaca-cleaned` 前 200 条平均 181 token、
   最长 454。`--seq-len 256` 会截断约 1/4 的响应。

## 实测结果

配置：`Qwen/Qwen3.5-0.8B`、`yahma/alpaca-cleaned` 前 512 条、`seq_len 256`、
`batch 1`、`fp32`、4 线程、LoRA `r=8 α=16`。

| 指标 | 值 |
|---|---|
| 主干参数 | 852,985,920（852.99 M） |
| 可训练参数 | 5,111,808（**0.599%**） |
| forward / backward | 8.56 s / 10.13 s（170 token 的样本） |
| 每步耗时 | **13.3 s**（范围 3.6–19.9 s，与 token 数近似线性） |
| 吞吐 | **12.6 token/s** |
| 峰值 RSS | **3.15 GiB**（16 GB 物理内存的 20%） |
| 20 步总耗时 | 4.4 min |

**结论：这台机器训 0.8B 的瓶颈是时间，不是内存。**
峰值内存只用了 3.15 GiB，而每步要 13.3 秒。
按每步消耗 1 条样本算，在完整 51,760 行的 alpaca 上跑满 1 个 epoch 约需 **7.9 天**。

两条性能线索（都实测到，未进一步优化）：

1. `causal_conv1d` 与 `chunk_gated_delta_rule` 都**回退到参考 PyTorch 实现**。
   这两个是 GDN 线性注意力的优化内核，只在 CUDA 上有实现，CPU 上装不了
   （依赖 triton）。也就是说 18/24 层的注意力走的都是未优化的纯 PyTorch 路径。
2. 每步耗时与 token 数近似线性：`t ≈ 2.2 s + 0.056 s × token`。
   `seq_len 256` 会被截断约 1/4 的响应（512 条切片里最长 454 token），
   所以调大 `seq_len` 会更慢但更完整。

可复现命令见上方「用法」。逐 step 明细在 `qlora_lab/runs/alpaca_20step/metrics.json`，
图示在 `qlora_lab/reports/alpaca_20step.html`。
