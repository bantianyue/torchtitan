# torchtitan 项目长期记忆

## 用户约束（必须遵守）
- `D:\02_source\torchtitan\torchtitan` 下**只能在 `models/` 新增模型目录**，
  **禁止修改其他目录的任何代码**。
- CPU 兼容性一律放在 `cpu_lab/`（仓库根下的兄弟目录），用**运行时 monkeypatch** 实现。
- **所有产物只放 D 盘，禁止往 C 盘写**。C 盘仅剩 ~11 GiB。
  `uv` 默认缓存 `%LOCALAPPDATA%\uv\cache`、`huggingface_hub` 默认 `~/.cache/huggingface`
  —— **两个默认值都在 C 盘，任何新脚本都必须显式设 `UV_CACHE_DIR` 和 `HF_HOME` 到 D 盘**。
- **禁止长时间无响应**：长任务必须放后台并**分段汇报**。

## 本机环境
- **唯一正确的解释器：`D:\02_source\torchtitan\.venv\Scripts\python.exe`**
  = Python 3.11.15 + torch 2.12.1+**cpu**，无 CUDA，只有 gloo。uv venv，
  `include-system-site-packages = false`。
  别误用 `C:\Users\twfehh7\AppData\Local\hermes\...` 或 managed python（都没有 torch）。
- 8 核。8 进程 CPU 分布式可跑。
- **内存**：物理 16 GiB，**commit 上限 45.63 GiB**（受 C: 剩余空间限制）。
  Windows 上 **物理内存才是硬上限**，commit 不是硬数；`avail_phys` 常年 3–5 GiB。
  **桌面基线常驻 ~25–30 GiB commit**（大户：WindowsTerminal ~3.2、chrome ×6 ~3.6、WorkBuddy ~1.7）。
- `cpu_lab/run_cpu_train.py` 自带 PYTHONPATH、用 `sys.executable` 拉子进程、
  从 `__file__` 推导仓库根 → **异目录 + 全绝对路径**也能跑（已实测从 `C:\` 成功）。
- **工具坑**：Bash 的 PATH 损坏；PowerShell stdout 不回传 → **结果写文件再 Read**。

## 新增模型的标准接法（不碰 `models/__init__.py`）
- `--module torchtitan.models.<name>`；`config/manager.py` 会自动尝试
  `importlib.import_module(f"{module}.config_registry")`，无需注册进 `_supported_models`。
- 必须提供：`model.py`、`sharding.py`（`set_<name>_sharding_config`）、`parallelize.py`、
  `__init__.py`（`model_registry` 返回 `ModelSpec`）、`config_registry.py`。
- `ModelSpec` 无 sharding 字段，分片靠各 `Module.Config.sharding_config`。

## CPU 上运行 torchtitan 的必修补丁（详见 `cpu_lab/cpu_compat.py`）
1. triton 无 CPU wheel → meta path finder 提供 permissive 桩；且
   `models/common/attention.py` **模块级**调 `torch.compile` → 把 `torch.compile`
   降级为 eager 透传（否则 inductor 需要真 triton）。
2. `tools/utils.py` 的 `_get_available_device_type() or "cuda"` 在 CPU 上拿不到 `"cpu"`
   → **导入 torchtitan 前**包住它。同时修好 DeviceMesh 与 gloo backend 选择。
3. Windows CPU torch 无 libuv，`TCPStore` 必须显式 `use_libuv=False`（环境变量无效）
   → **torchrun 不可用**，自己 spawn 子进程设 `RANK/LOCAL_RANK/WORLD_SIZE`。
4. `barrier(device_ids=[torch.cpu.current_device()])` 传字符串 `"cpu"` 报错；
   `torch.distributed.set_timeout` 在 2.12 已移除。→ 都要补。
5. `DeviceMemoryMonitor` 依赖 `torch.cuda` API → 换成 host-RSS 实现（接口同形）。
6. FSDP 梯度除法用 NCCL 专有 `ReduceOp.PREMUL_SUM`，gloo 报错 →
   FSDP 包完模型后对所有模块 `set_force_sum_reduction_for_comms(True)`。
7. **PP 的 step API 不匹配**：torch 2.12 的 `step()` 是整批入口，torchtitan 传 `arg_mbs=`
   → 落进 `**kwargs` 被当模型输入。microbatch 级入口 `_step_microbatches(arg_mbs, ...)`
   签名正好匹配 → 包 `pipeline_parallel._build_pipeline_schedule`，给返回的 schedule
   实例装 `step` 适配器；`loss_kwargs` 绑定进 loss fn 的 holder。
   `_prepare_schedule_with_comms` 各 schedule 在 `__init__` 已调用，无需重放。

## 并行策略在本环境的可用性（已实测）
- 可用：single / fsdp / tp（含 SP）/ tp_fsdp / tp_sp / pp / pp_tp / pp_fsdp
- 需条件：pp_tp_fsdp 必须 `--sp false`；hsdp 在 4 rank（2×2）可用，8 rank（2×4）挂
- 不可用：cp 系（CPU SDPA 派发到
  `_scaled_dot_product_flash_attention_for_cpu`，无 DTensor 分片策略）
  → PP×TP×FSDP×CP 四维组合本机不可达
- 装不下而非逻辑错：Qwen2.5-0.5B 的 `pp_fsdp`×4（TinyGPT 同组合能跑完 ⇒ 逻辑没问题），
  需 ~20 GiB commit 而宿主只剩 ~20 且物理 RAM 只余 ~5 ⇒ 原生崩溃，见下方修正记录

## 校验并行正确性的判据
TP 不对 batch 做分片 → **所有 rank 的 loss 必须逐位相同**。不一致说明漏了集合通信。
反之 FSDP 会分片 batch，跨策略比较 loss 前必须先对齐 global batch。

## 容易误判为「训练 bug」的坑
- **多进程同时准备 HF dataset 会争锁挂住**，其他 rank 随后报 gloo 超时。
  先看日志里**哪些 rank 没到 `Training starts`**再下结论。
- PP 下只有最后一个 stage 记真实 loss，其余记 `-1` 哨兵；`grad_norm` 全 rank 相同。

## 缩放阶梯（TinyGPT，2026-10-12 实测）
策略固定 `pp2×tp2×dp_shard2`（8 rank, `--sp false`），seq 128 / local batch 8。
27.5M → 6.5 s/step；88.5M → 15.0；206M → 36.5；**687M 崩在 FSDP all-gather 分配**。
拟合：commit(合计) = 41.2 B/param × N + 4.08 GiB；RSS = 20.3 B/param × N + 3.35 GiB；
每步 = 0.170 s/M + 1.1 s。
工具：`cpu_lab/scaling.py`（>200M 必须加 `--no-warmup`）→ `cpu_lab/reports/scaling.html`。

## Qwen3.5-0.8B 用 torchtitan 在 CPU 上跑通（2026-10-12 晚）

### 三条最有价值的结论
1. **torchtitan 上游就带完整 Qwen3.5 实现**（`torchtitan/models/qwen3_5/`，含 `0.8B`
   flavor + vision tower）。**不需要写模型代码**，只需要 CPU 兼容层
   `cpu_lab/qwen35_cpu.py`（monkeypatch，6 个补丁：fla 桩 / eager flex_attention /
   跳过 BlockMask 构建 / causal_conv1d_varlen / 补 text-only special tokens /
   `os.path.isabs` 取代 POSIX-only 路径校验）。
2. **必须开 weight tying**：HF `tie_word_embeddings=true` ⇒ 磁盘上**没有**
   `lm_head.weight` ⇒ 不开时 `dcp.load` 报 `Missing key ... lm_head.weight`；
   开了还省 254M 参数。`_0_8b` 里「torchtitan 不支持 tied embedding」的注释**已过时**。
3. **`training.dtype` ≠ `mixed_precision_param`**：前者在 meta 构建期
   `torch.set_default_dtype`，使参数/梯度/Adam **全部**为该 dtype（无 fp32 主副本）；
   后者只改前向/通信 dtype。**要省内存必须用前者**（fp32 13.65 GB → bf16 6.83 GB）。

### 参数与实测
- 参数：dense 752,393,024 + vision 100,592,896 = **852,985,920**（tied）。
  24 层 = 18 GatedDeltaNet + 6 full attention。
- HF→DCP：594 张量 / 4.43 GB → `cpu_lab/checkpoints/qwen35_0_8b_hf`；
  丢弃的 15 个键全是 `mtp.*`。
- **单 rank bf16 / seq128 / batch1：64 s/step，loss 12.81853 → 12.84252，
  grad_norm 91 → 87。峰值 RSS 6.47 GiB / commit 11.68 GiB。**
- **真实权重 SFT（`qwen35_0_8b_sft_bf16`）：10 步，loss 1.33863 → 0.24605，
  grad_norm 145 → 41.25，wall 819 s，峰值 RSS 7.69 GiB / commit 11.78 GiB。**
  产物 `cpu_lab/outputs/qwen35/checkpoint/step-10/`（2.22 GB）。
  step 1 就 ~1.3（随机初始化是 12.8）⇒ 转换与 tying 都正确。

### 多 rank：根因是集合通信超时，不是内存（重要修正）
`Trainer.Config.comm` 默认 **init 300 s、之后收紧到 100 s**——按加速器定的。
但**分页的 CPU rank 一次 all-gather 就要几分钟**，「首步后收紧到 100 s」等于保证超时。

| rank 数 | 合计峰值 commit | 系统低水位(commit 余量) | 每步耗时 | 结果 |
|---|---|---|---|---|
| 1 | 11.68 GiB | 4.61 GiB | 64 s | 完成 |
| 2 | 16.32 GiB | 2.08 GiB | ~140 s | 默认超时**崩**；1800 s**完成** |
| 4 | 18.88 GiB | 0.47 GiB | — | `access violation`（两种超时都崩） |

崩时 commit 还剩 2.08 GiB ⇒ **不是内存耗尽**。1 rank 能过不是因为装得下
（它同样把 `avail_phys` 压到 0.01 GiB），而是**它没有集合通信，没人给它超时**。

**内存模型**：总 commit ≈ 模型状态常量（bf16 时 6.83 GiB）+ 每 rank 不可分片开销 × rank 数。
纯 FSDP 下每 rank 开销 ≈ 4.8 GiB（all-gather 输出 1.7 + 展平梯度缓冲 1.7 + 运行时 ~1.4）
⇒ 8 rank 纯 FSDP ≈ 45 GiB，顶到 45.63 GiB 上限。
**PP/TP 会真正缩小每 rank 缓冲**（pp2×tp2 ⇒ 缓冲降到 1/4，每 rank ≈ 2.3 GiB）
⇒ 8 rank pp2×tp2×dp2 ≈ 25 GiB。

**改法**：`CommConfig(init_timeout_seconds=1800, train_timeout_seconds=1800)`。
已加配置 `qwen35_0_8b_bf16_multi`。这是**诊断手段不是生产设置**：
放宽后能跑完 ⇒ 「装得下但在换页」；放宽后仍卡住 ⇒ **先确认是否还有 rank 活着**——
Qwen2.5 pp_fsdp 的「死锁」最后查明是某 rank 原生崩溃（access violation / oneDNN
原语失败 / SIGABRT 三种死法）、幸存者等一个已死的 peer，超时对死者无效。
判据：看哪个 rank 的日志里有原生崩溃签名，别看第一个非零退出码。
**Windows 上真正的硬墙是 commit 上限耗尽**（无 catchable OOM，崩在 whoever
allocated last）；物理 RAM 耗尽只是换页，能活（pp4 实测 avail_phys 0.00 仍跑完，
pp_fsdp commit 余 0.05 GiB 就崩）。

**无效手段（已测）**：`PYTORCH_CPU_ALLOC_CONF=expandable_segments:True` 在
torch 2.12.1 上**被静默接受但无效**（仅 CUDA）。分配 churn A/B 实测无差异。

### 踩坑
- **`ChatDataset` 的 shuffle 缓存路径与 rank 无关**：`shuffle(seed=42)` 在所有 rank 上
  产生**同一个** permutation hash ⇒ 8 rank 抢写同一 `cache-*.arrow`，rank 1 报
  `OSError: [Errno 22]`，其余卡在数据集准备。
  解法：先跑 `cpu_lab/prewarm_dataset.py`（单进程、不建模型）建缓存，再 `--no-warmup`。
  **不是 MAX_PATH 问题**（路径 173 字符、LongPathsEnabled=1）。
- 模块门控写错**不会响**：`cpu_lab_child.py` 曾判 `"qwen3_5" in module`，
  而模块名是 `cpu_lab.qwen35` ⇒ 补丁没装 ⇒ `config_registry` 导入 `fla` 失败 ⇒
  `ConfigManager` 的 `except ImportError: continue` **静默吞掉**。
  只读 `sys.argv` 的门控要同时匹配两种拼写。

## qlora_lab：HF 生态的微调（torchtitan 之外）
- torchtitan **做不了** HF 模型的微调（无权重加载/PEFT/量化）；这类工作放
  `qlora_lab/`（与 `cpu_lab/` 平级），**独立 venv**，不改 torchtitan 代码。
- venv `qlora_lab/.venv`：torch 2.14.1+cpu、transformers 5.19.0、peft 0.21.2、
  accelerate 1.15.0、datasets 5.1.0、psutil。**Qwen3.5 需 transformers ≥ 5.2.0**。
- 实测（4 核 CPU）：**13.3 s/step、12.6 token/s、峰值 RSS 3.15 GiB**（seq256/batch1/fp32），
  LoRA r=8 可训练 5,111,808（0.599%）。**瓶颈是时间不是内存**。
- **CPU 上跑 HF 训练的三个必设项**：
  1. `attn_implementation="eager"` —— CPU SDPA 无反向 kernel；
  2. `enable_input_require_grads()` —— 否则冻结 embedding 下游的 LoRA 拿不到梯度；
  3. 接受 `causal_conv1d` / `chunk_gated_delta_rule` 回退到参考实现（优化内核只在 CUDA）。
- **网络**：直连 huggingface.co 不可用；用 `hf-mirror.com`。
  `snapshot_download` 的 Xet 后端会卡死 ⇒ `HF_HUB_DISABLE_XET=1`；Windows 无符号链接权限 ⇒ 不续传。
  大文件用 `fetch_model_mirror.sh`（8 连接 Range 并行 + sha256 校验）。
- **CPU 上 4bit（QLoRA）是负优化**——反量化比原生 fp32 更慢，0.8B fp32 才 ~3.4 GB。
  省内存要靠 LoRA（主干冻结），不是靠量化。

## 观测与验证工具（cpu_lab/）
一条命令跑完「训练 + 通信采集 + 推理验证 + HTML 报告」：
```
python cpu_lab/run_cpu_train.py --combo <preset> -n 8 --collectives --probe \
    --dataset-path cpu_lab/planted --steps N [--sp false] [--log-dir DIR]
```
产出 `cpu_lab/reports/<combo>.html`（自包含、无 CDN）。`report.py --log-dir D --combo C`
可用已有日志重出报告。`--log-dir` 让多次实验产物并存（默认 `cpu_lab/logs` 会被覆盖）。
模块：`collectives.py`、`planted_corpus.py`、`probe_inference.py`、`report.py`。

**测量集合通信必须两层**：dispatcher + Python 包装 + `P2POp.__new__`。
FSDP 走 Python 的 `dist.*` 且会再派发到 aten（需去重）；`fsdp.all_gather_copy_in`
是本地拷贝（须排除）；PP 的 P2P 走 `isend/irecv` 快路径（不能包装，会破坏
`P2POp` 的 identity 检查）；PP 的 grad-norm 走纯 Python `all_reduce`；
**单卡运行时唯一的集合通信来自存 checkpoint**（DCP 用 broadcast 做 object scatter）
⇒ 必须按 phase（step/init/checkpoint）分类。不知道谁发的就用
`CPU_LAB_COLLECTIVES_STACK=1` 记调用栈，别猜。

**推理验证的两个坑**（都会误判为「权重没训好」）：
1. 不要无条件把 `lm_head` 重绑到 embedding —— PP 用的 **untied** flavor 下会把训练好的
   输出头换成 embedding，输出变噪声。
2. prompt 必须带 **BOS** 并与训练的 position 对齐。

## 启动器（`run_cpu_train.py`）的两处改动
- 内存采样器新增**运行期系统最低水位**（`system_low`：`min_avail_phys` /
  `min_commit_free` / `max_commit_used`），渲染进 `summary.txt`。
  理由：per-rank 峰值合计只回答「这个 job 多大」，回答不了「它装得下吗」——
  宿主是共享的，同一个 job 高基线会死、低基线会活。**加了这个指标当场定位了根因。**
- 失败路径也要写 `run_json`/`summary.txt`（此前崩溃时内存数据被整段丢弃，
  而那恰好是唯一需要看的那次）。

## 换机器：产物与说明已推送 GitHub（2026-10-12 末）

**方案：只提交无法重建的东西。** venv / HF 权重 / DCP checkpoint / 日志一律不进 Git，
交由 `cpu_lab/bootstrap.py` 重建。克隆 ~150 MB 而非 ~7 GB。

- 远端：`https://github.com/bantianyue/torchtitan.git`（`origin/main`，已同步）。
- `cpu_lab/bootstrap.py`：6 步 `venv → deps → assets → dcp → data → smoke`，
  幂等、可 `--steps` / `--skip`，失败打印续跑命令。
- `cpu_lab/MIGRATION.md`：产物清单 / 重建步骤 / 实测基准 / 策略矩阵 / 已知坑。
- 冻结依赖：`cpu_lab/requirements-cpu.txt`、`qlora_lab/requirements.txt`。
  **venv 是 uv 建的、没有 pip** ⇒ 必须用 `uv pip freeze --python <py>`，
  且要剔除 `-e file:///D:/...` 这类旧机器绝对路径。
- `.gitignore` 已排除 `.venv/`、`cpu_lab/checkpoints/`、`qlora_lab/{hf,model,runs}/`、
  `*.safetensors`、`*.distcp`、`cpu_lab/logs*/`、`cpu_lab/_*.txt`。

**不用 Git LFS**：免费 10 GiB 存储 / 10 GiB 月流量，权重就 6.6 GiB，且比镜像站重下慢。
**备选**：物理可达时用 `robocopy` 整目录直传；但 Windows venv 含绝对路径，**仍要重建**。

## 换 32 GiB 机器的结论（2026-10-12 末）
- **内存**：够。pp2×tp2×dp2 需 ~25 GiB；32 GiB 物理下基线几乎不换页，可用 20–24 GiB。
  纯 FSDP 8 rank 需 ~45 GiB，仍然贴上限 ⇒ **8 rank 必须带 PP/TP，不能纯 dp_shard**。
- **速度**：不划算。实测 1 rank 64 s/step、2 rank ~140 s/step（慢 2.2×）。
  8 核上 rank 数超过物理核数 = 纯亏（gloo/TCP 每次 all-gather 1.7 GiB）。
- ⇒ 32 GiB 的正确用法是 **4 rank（pp2×dp2 或 dp_shard4）+ 每 rank 2 线程**，而非 8 rank。
