# 换机器重跑：完整说明

本文档回答两个问题：**仓库里有什么**，以及**换台机器之后怎么把它全装回来**。

核心原则只有一条：**进 Git 的只有无法重建的东西。**
venv、HF 权重、DCP checkpoint、日志全是派生品，重下/重算比提交更快，也更不容易过期。
克隆整个仓库约 **150 MB**，而不是 **7 GB**。

---

## 一、仓库里有什么 / 没有什么

### 已提交（真正的资产）

| 路径 | 内容 |
|---|---|
| `cpu_lab/*.py` | CPU 兼容层与实验工具链（`cpu_compat.py`、`qwen35_cpu.py`、`run_cpu_train.py`、`scaling.py`、`collectives.py`、`probe_inference.py`、`report.py` …） |
| `cpu_lab/qwen35/` | Qwen3.5 的 `config_registry.py`（各实验配置） |
| `cpu_lab/bootstrap.py` | **一键重建脚本**（本文档的主角） |
| `cpu_lab/requirements-cpu.txt` | 冻结的精确依赖版本（78 个包） |
| `cpu_lab/reports/*.html` | 自包含 HTML 报告（无 CDN，可离线打开） |
| `cpu_lab/planted/data.json` | 植入暗号语料 |
| `qlora_lab/*.py`, `*.sh` | HF 生态微调脚本 |
| `qlora_lab/requirements.txt` | 冻结的 qlora 环境（57 个包） |
| `torchtitan/models/tinygpt/` | 缩放阶梯实验用的测试模型 |
| `.workbuddy/memory/` | 项目记忆与踩坑记录（**含全部实测数据**） |
| `cpu_lab/README.md` | 工具链说明 |

### 未提交（需重建）

| 路径 | 大小 | 重建方式 |
|---|---|---|
| `.venv/` | 0.85 GiB | `bootstrap.py` 的 `venv` + `deps` 步 |
| `qlora_lab/.venv/` | 0.84 GiB | `qlora_lab/setup_env.sh` |
| `qlora_lab/model/` | 1.65 GiB | `assets` 步：从 `hf-mirror.com` 下载 |
| `qlora_lab/hf/` | 0.92 GiB | 下载缓存（含 0.79 GiB 未下载完的残片，无需保留） |
| `cpu_lab/checkpoints/` | 4.13 GiB | `dcp` 步：`convert_qwen35_hf.py` |
| `cpu_lab/logs*/` | — | 运行时产物；关键数字已提炼进本报告与 `reports/` |

> **为什么不用 Git LFS**：GitHub 免费额度是 10 GiB 存储 / 10 GiB 月流量，
> 光权重就 6.6 GiB，一次上传就吃掉大半，而且比从镜像站重下还慢。

---

## 二、新机器上的操作

### 快速开始（推荐）

```bash
# 1. 克隆到剩余空间充足的盘（不必是 D 盘，脚本全部使用仓库内相对路径）
git clone https://github.com/bantianyue/torchtitan.git
cd torchtitan

# 2. 一键重建：venv -> 依赖 -> HF 权重 -> DCP -> 数据集 -> smoke 验证
python cpu_lab/bootstrap.py
```

`bootstrap.py` 会按顺序执行 6 步，每步幂等，失败会打印**从失败步继续**的命令：

```bash
python cpu_lab/bootstrap.py --steps dcp,data,smoke   # 只跑这几步
python cpu_lab/bootstrap.py --skip assets            # 权重已就位时跳过下载
```

各步作用：

| 步 | 做什么 | 产出 |
|---|---|---|
| `venv` | 用 uv 建 CPython 3.11 的 `.venv` | `.venv/` |
| `deps` | **先从 PyTorch CPU 索引装 torch**，再从 PyPI 装其余 77 个包 | 环境就绪 |
| `assets` | 从 `hf-mirror.com` 抓 `Qwen/Qwen3.5-0.8B` | `qlora_lab/model/` |
| `dcp` | HF safetensors → torchtitan DCP 布局 | `cpu_lab/checkpoints/qwen35_0_8b_hf/` |
| `data` | 生成植入语料 + **单进程**预热数据集缓存 | `cpu_lab/planted/`、HF 缓存 |
| `smoke` | 单 rank 跑 2 步（随机初始化） | `cpu_lab/logs_qwen35/smoke/` |

> `deps` 分两阶段是必须的：`requirements-cpu.txt` 里钉了 `torch==2.12.1`，
> 直接从 PyPI 装在 Windows 上会拉到 ~2.5 GB 的 **CUDA** 构建。
> CPU 索引必须优先于 torch，其余包仍走 PyPI。

### 手动版（想自己控制每一步）

```bash
uv venv .venv --python 3.11

uv pip install --python .venv/Scripts/python.exe \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.12.1 torchdata==0.11.0

uv pip install --python .venv/Scripts/python.exe \
    -r cpu_lab/requirements-cpu.txt   # 已剔除 torch 行

uv pip install --python .venv/Scripts/python.exe -e .   # torchtitan 用本工作树

python cpu_lab/convert_qwen35_hf.py
python cpu_lab/prewarm_dataset.py --hf-cache cpu_lab/hf
python cpu_lab/run_cpu_train.py --combo qwen35 -n 1 \
    --module cpu_lab.qwen35 --config qwen35_0_8b_smoke \
    --hf-cache cpu_lab/hf --show-logs
```

Windows 下用 `.venv\Scripts\python.exe`，Linux 下用 `.venv/bin/python`。

---

## 三、实测基准（用于判断新机器够不够）

模型：**Qwen3.5-0.8B，852,985,920 参数**（dense 752,393,024 + vision 100,592,896，
已开 weight tying）。24 层 = 18 层 GatedDeltaNet + 6 层全注意力。

### 单 rank 参考结果

```
配置：bf16 / seq_len 128 / batch 1 / lr 5e-5 / warmup 2
step:  1  loss: 1.33863  grad_norm: 145.0000
step: 10  loss: 0.24605  grad_norm:  41.2500
wall 819.1 s，峰值 RSS 7.69 GiB / commit 11.78 GiB
```

step 1 就到 ~1.3（随机初始化是 12.8）⇒ 权重转换与 tying 都正确。

### rank 数与内存

| rank 数 | 合计峰值 commit | 系统低水位(commit 余量) | 每步耗时 | 结果 |
|---|---|---|---|---|
| 1 | 11.68 GiB | 4.61 GiB | **64 s** | 完成 |
| 2 | 16.32 GiB | 2.08 GiB | **~140 s** | 默认超时崩；1800 s 完成 |
| 4 | 18.88 GiB | 0.47 GiB | — | `access violation` |
| 8 | ~45 GiB（外推） | — | — | 崩在 FSDP 应用后 |

### 内存模型（换机器后可直接套用）

```
总 commit ≈ 模型状态常量 + 每 rank 不可分片开销 × rank 数

模型状态常量（852.99M 参数）
    fp32: 3.41(参数) + 3.41(梯度) + 6.83(Adam) = 13.65 GB
    bf16: 1.71 + 1.71 + 3.41                  =  6.83 GB

纯 FSDP 每 rank 不可分片开销 ≈ 4.8 GiB @bf16
    ├─ all-gather 输出缓冲   1.71 GiB（全量未分片参数）
    ├─ 展平梯度缓冲           1.71 GiB（reduce-scatter 用）
    └─ Python + torch 运行时  ~1.4 GiB
```

**关键区分**：`dp_shard` **不缩小**上面那两项缓冲，所以纯 `dp_shard=8` ≈ 45 GiB，
32 GiB 机器上仍然贴顶。而 **PP/TP 会真正缩小**——`pp2×tp2` 下每 rank 的 FSDP
单元只剩 1/4 层，缓冲降到 0.43 + 0.43 GiB：

- `pp2×tp2×dp_shard2`（8 rank）≈ **25 GiB** → 32 GiB 机器可行
- 纯 `dp_shard=8`（8 rank）≈ **45 GiB** → 仍然危险

⇒ **8 rank 必须带 PP/TP 维度，不能只用 `dp_shard=8`。**

### 速度：这才是真正的判据

2 个 rank 比 1 个 rank 慢 2.2 倍。原因是 gloo over TCP 每次 all-gather 要搬
1.71 GiB，8 个进程还要在同一台机器的 8 个核上互相抢。
**多开 rank 增加的是通信，不是算力。**

⇒ 32 GiB 机器的正确用法是 **4 rank（`pp2×dp_shard2` 或 `dp_shard4`）+ 每 rank 2 线程**，
而不是 8 rank × 1 线程。投资顺序：**更多物理核 > 更多内存 > 更多 rank**。

---

## 四、并行策略可用性矩阵

| 策略 | 状态 |
|---|---|
| `single` / `fsdp` / `tp` / `tp_fsdp` / `tp_sp` / `pp` / `pp_tp` / `pp_fsdp` | ✅ 可用 |
| `pp_tp_fsdp` | ⚠️ 必须 `--sp false` |
| `hsdp` | ⚠️ 4 rank（2×2）可用；8 rank（2×4）挂 |
| `cp` / `tp_cp` | ❌ CPU SDPA 派发到 `_scaled_dot_product_flash_attention_for_cpu`，无 DTensor 分片策略 |

⇒ 四维组合 PP×TP×FSDP×CP 在 CPU 上不可达；去掉 CP 的三维组合可跑。

---

## 五、已知坑（换机器后照方抓药）

### 1. 多 rank 崩溃的真因是集合通信超时，不是内存

`Trainer.Config.comm` 默认 **init 300 s、之后收紧到 100 s**。这是按加速器定的：
100 s 没完成就是挂了。但分页的 CPU rank 一次 all-gather 合法地要几分钟，
「首步后收紧到 100 s」等于保证后续每步都超时。

```python
CommConfig(init_timeout_seconds=1800, train_timeout_seconds=1800)
# 或 CLI: --comm.init_timeout_seconds 1800 --comm.train_timeout_seconds 1800
```

已内置配置 `qwen35_0_8b_bf16_multi`。**这是诊断手段不是生产设置**：
放宽后能跑完 ⇒ 「装得下但在换页」；1800 s 仍超时 ⇒ 真死锁。

判断依据：崩溃时 commit 还余 **2.08 GiB**（2 rank），说明不是内存耗尽。

### 2. 数据集 shuffle 缓存路径与 rank 无关

`shuffle(seed=42)` 在所有 rank 上产生**同一个** permutation hash ⇒ 8 个 rank
抢写同一个 `cache-*.arrow`，rank 1 报 `OSError: [Errno 22]`。

解法：先跑 `prewarm_dataset.py`（单进程、不建模型）建好缓存，再 `--no-warmup` 起多 rank。
**不是 MAX_PATH 问题**（路径 173 字符、`LongPathsEnabled=1`）。

### 3. 模块门控拼写不会报错

`cpu_lab_child.py` 里判 `"qwen3_5" in module`，而模块名是 `cpu_lab.qwen35`（无下划线）
⇒ 补丁没装 ⇒ `config_registry` 导入 `fla` 失败 ⇒ `ConfigManager` 的
`except ImportError: continue` **静默吞掉**，最后只报「找不到配置函数」。

只读 `sys.argv` 的门控要同时匹配两种拼写。

### 4. `training.dtype` ≠ `mixed_precision_param`

前者在 meta 构建期 `torch.set_default_dtype`，使参数/梯度/Adam **全部**为该 dtype；
后者只改前向/通信 dtype。**要省内存必须用前者。**

### 5. 必须开 weight tying

HF `tie_word_embeddings=true` ⇒ 磁盘上**没有** `lm_head.weight` ⇒ 不开时
`dcp.load` 直接报 `Missing key in checkpoint state_dict: lm_head.weight`。
（`_0_8b` 源码里「torchtitan 不支持 tied embedding」的注释**已过时**。）

### 6. 无效手段（别再试）

`PYTORCH_CPU_ALLOC_CONF=expandable_segments:True` 在 torch 2.12.1 上
**被静默接受但无效**（该选项仅 CUDA）。分配 churn A/B 实测无差异。

### 7. Windows 专属

- torch CPU 构建无 libuv ⇒ `TCPStore` 必须显式 `use_libuv=False`
  ⇒ **torchrun 不可用**，要自己 spawn 子进程设 `RANK/LOCAL_RANK/WORLD_SIZE`。
- `initial_load_path` 校验是 POSIX-only（`startswith("/")`），拒 `D:\...`。
- 这两条在 Linux 上都不需要；`cpu_compat` 内部已按平台门控。

---

## 六、常用命令

```bash
# 并行策略一览
python cpu_lab/run_cpu_train.py --list

# 缩放阶梯（TinyGPT，找当前机器的模型规模上限）
python cpu_lab/scaling.py --log-dir cpu_lab/logs_scale/<flavor>   # >200M 加 --no-warmup

# Qwen3.5 单 rank 真实权重 SFT
python cpu_lab/run_cpu_train.py --combo qwen35 -n 1 \
    --module cpu_lab.qwen35 --config qwen35_0_8b_sft_bf16 \
    --hf-cache cpu_lab/hf --no-warmup

# 多 rank（必须放宽超时）
python cpu_lab/run_cpu_train.py --combo qwen35 -n 4 \
    --module cpu_lab.qwen35 --config qwen35_0_8b_bf16_multi \
    --hf-cache cpu_lab/hf --no-warmup --stagger 5

# 全流程：训练 + 通信采集 + 推理验证 + HTML 报告
python cpu_lab/run_cpu_train.py --combo <preset> -n 8 --collectives --probe \
    --dataset-path cpu_lab/planted --steps N [--sp false] [--log-dir DIR]

# 用已有日志重出报告（不重训）
python cpu_lab/report.py --log-dir D --combo C
```

`--log-dir` 用来让多次实验产物并存（默认 `cpu_lab/logs` 会被下次运行覆盖）。

---

## 七、产物目录

```
cpu_lab/
├── bootstrap.py              # 一键重建（换机器后第一条命令）
├── requirements-cpu.txt      # 冻结依赖
├── MIGRATION.md              # 本文档
├── README.md                 # 工具链说明
├── cpu_compat.py             # CPU 运行时补丁（7 处）
├── qwen35_cpu.py             # Qwen3.5 的 CPU 兼容层（6 处补丁）
├── run_cpu_train.py          # 多 rank 启动器
├── scaling.py                # 缩放阶梯
├── checkpoints/              # [重建] HF→DCP 权重
├── outputs/                  # 训练 checkpoint 与结构化日志
├── reports/                  # HTML 报告（已提交）
├── planted/                  # 植入暗号语料（已提交）
├── hf/                       # [重建] HF 缓存
└── logs*/                    # [重建] 运行日志

qlora_lab/
├── requirements.txt          # 冻结依赖（已提交）
├── *.py / *.sh               # 微调脚本（已提交）
├── model/                    # [重建] HF 权重 + tokenizer
├── hf/                       # [重建] 下载缓存
└── runs/                     # [重建] 训练产物
```

---

## 八、网络备注

- `huggingface.co` **直连不可用**（curl 返回 `000`），必须走 `hf-mirror.com`。
  `bootstrap.py` 会自动设 `HF_ENDPOINT`。
- `snapshot_download` 的 **Xet 后端会卡死**，必须 `HF_HUB_DISABLE_XET=1`。
- Windows 无符号链接权限 ⇒ 不支持断点续传；大文件用
  `qlora_lab/fetch_model_mirror.sh`（8 连接 Range 并行 + sha256 校验）。
