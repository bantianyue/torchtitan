# 换机器交接说明 — 纯 CPU 上用 torchtitan 训练 Qwen3.5-0.8B

本文件是**给新机器上的 WorkBuddy 看的第一份文档**。目标是让一个没有上下文的会话
在 10 分钟内回答三个问题：**这件事在做什么、做到哪了、下一步该跑什么命令。**

上手只需两条命令：

```bash
git clone https://github.com/bantianyue/torchtitan.git
cd torchtitan && python cpu_lab/bootstrap.py
```

跑完之后照「五、验证方法」逐层验收，再决定「六、后续要怎么做」里的哪条路线。

---

## 一、任务目标

### 1.1 原始需求

在一台**没有 GPU** 的 Windows 机器上，用 **torchtitan 框架**训练 **Qwen3.5-0.8B**。
（最初的说法是「4bit 量化训练」，实测结论见 6.5 —— 那条路在 CPU 上是负优化，已放弃。）

### 1.2 硬约束（不可违反）

| 约束 | 说明 |
|---|---|
| **必须用 torchtitan** | 不允许换成 HF Trainer / LLaMA-Factory 之类 |
| **不得修改 `torchtitan/`** | 一行都不改。新增模型只能放进 `torchtitan/models/<新目录>/` |
| **所有产物落 D 盘** | C 盘只剩 ~11 GiB。HF 缓存、uv 缓存、输出目录全部显式指定到非系统盘 |

CPU 兼容性因此**全部实现为运行时 monkeypatch**，放在仓库根下的 `cpu_lab/`（与 `torchtitan/`
平级的兄弟目录）。`torchtitan/` 至今零改动。

### 1.3 这一阶段实际在回答的问题

1. torchtitan 的 Trainer / FSDP / TP / PP 能不能在 CPU-only torch 上真跑起来？
2. 这台机器能跑多大的模型？（缩放阶梯 + 内存模型）
3. 哪些并行策略组合可达，哪些不可达？

---

## 二、当前进展

> 状态截至 **2026-10-12**，仓库 HEAD = `aa7406eef`。

### 2.1 已完成

| # | 成果 | 证据 |
|---|---|---|
| 1 | 发现 **torchtitan 上游自带完整 Qwen3.5 实现**（`torchtitan/models/qwen3_5/`，含 `0.8B` flavor + vision tower） | 源码在仓库里，无需写模型代码 |
| 2 | 建成 **CPU 兼容层** `cpu_lab/qwen35_cpu.py`（6 个补丁：fla 桩、eager flex_attention、跳 BlockMask、causal_conv1d varlen、多模态特殊 token、Windows 路径校验） | 运行时 patch，`torchtitan/` 零改动 |
| 3 | 通用 CPU 兼容层 `cpu_lab/cpu_compat.py`（7 个补丁） | 同上 |
| 4 | **HF 权重 → torchtitan DCP 转换**完成：594 张量 / 4.43 GB，丢弃 15 个 `mtp.*` 键 | `cpu_lab/convert_qwen35_hf.py` |
| 5 | **单 rank 真实权重 SFT 10 步跑通** | `cpu_lab/reports/runs/sft_bf16_1r/` |
| 6 | **2 rank FSDP** 放宽集合通信超时后跑通，跨 rank loss / grad_norm 逐位一致 | `cpu_lab/reports/runs/bf16_multi_2r/` |
| 7 | 并行策略可用性矩阵测完（见 6.4） | `cpu_lab/README.md` |
| 8 | 代码 / 配置 / 文档 / 实测报告已提交推送 GitHub | commit `1ba67b15c` + `aa7406eef` |

**单 rank SFT 参考结果（最重要的验收基线）：**

```
配置：bf16 / seq_len 128 / batch 1 / lr 5e-5 / warmup 2
step:  1  loss: 1.33863  grad_norm: 145.0000
step: 10  loss: 0.24605  grad_norm:  41.2500
wall 819.1 s，峰值 RSS 7.69 GiB / commit 11.78 GiB
```

### 2.2 未完成 / 已知失败

| 项 | 状态 | 现象 |
|---|---|---|
| 4 rank FSDP | ❌ | `access violation`（崩在 FSDP 梯度 reduce-scatter） |
| 8 rank FSDP | ❌ | 日志停在 `Applied FSDP to the model` 后八个进程全消失，无 traceback |
| PP×TP×FSDP×CP 四维 | ❌ | CP 在 CPU 上不可达（见 6.4） |
| 4bit 量化训练 | ❌（主动放弃） | 见 6.5 |

### 2.3 ⚠️ 训练产物已丢失，必须在机器上重跑

`cpu_lab/outputs/qwen35/checkpoint/step-10/`（**2.22 GB**，上表那 10 步 SFT 的权重）
**已被后续多 rank 实验覆盖删除**。原因是 `run_cpu_train.py` 默认会在每次运行前清空
`cpu_lab/outputs/<combo>/`（否则 torchtitan 会尝试 resume，而 `last_save_model_only=True`
下缺少 dataloader 状态会直接报错）。

`summary.txt` 仍留有产物清单作为证据：

```
  150,089 B  checkpoint\step-10\.metadata
2,215,466,786 B  checkpoint\step-10\__0_0.distcp
```

⇒ 新机器上跑完 bootstrap 后，**第一件事是重跑那 10 步 SFT**（见 6.2），
并且**加 `--keep-outputs`**，别让下一次实验再把它清掉。

### 2.4 本轮（交接前）刚修的问题

| 问题 | 处理 |
|---|---|
| `bootstrap.py` 的 `main()` **忘了调用 `parse_args()`** —— 脚本此前从未真正可执行 | 已修；`--help` / `--steps` / `--hf-endpoint` 均正常 |
| `qlora_lab` 的 venv 只有 bash 脚本 `setup_env.sh` 能建，新机是 Windows 11 | 给 bootstrap 加了可选的 `qlora` 步骤（Python 实现，跨平台） |
| HF 镜像写死，新网络若可直连 huggingface.co 无法切 | 加 `--hf-endpoint` 参数 |
| 仓库根目录混入了 9 个早期会话的可视化草稿（`*_three_layers.html`、`local_map*.html`、`ep_three_layers.zip`）与 4 个 `.workbuddy/_env_probe*.txt` | 已确认只是草稿，无功能依赖；保留在仓库中不影响重建 |

---

## 三、仓库地图

### 进 Git 的（真正的资产，约 2.7 MB）

```
cpu_lab/
├── bootstrap.py            一键重建（新机器第一条命令）
├── requirements-cpu.txt    冻结依赖（78 个包）
├── MIGRATION.md            换机详细配方
├── README.md               工具链说明
├── cpu_compat.py           通用 CPU 运行时补丁（7 处）
├── qwen35_cpu.py           Qwen3.5 的 CPU 兼容层（6 处补丁）
├── run_cpu_train.py        多 rank 启动器（自己 spawn，不用 torchrun）
├── cpu_lab_child.py        子进程入口（安装补丁后再导入 torchtitan）
├── scaling.py              缩放阶梯
├── collectives.py          集合通信计数器
├── probe_inference.py      载入 checkpoint 做推理验证
├── report.py / qwen35_report.py   HTML 报告
├── prewarm_dataset.py      单进程数据集缓存预热
├── planted_corpus.py       植入暗号语料生成
├── qwen35/config_registry.py      Qwen3.5 的各档训练配置
├── planted/data.json       植入暗号语料（已提交）
└── reports/                HTML 报告 + runs/ 各轮实验摘要（已提交）

qlora_lab/                  HF 生态微调轨（脚本 + requirements.txt 已提交）
.workbuddy/memory/          项目记忆与全部实测数据（已提交）
```

### 不进 Git 的（约 7 GB，由 bootstrap 重建）

| 路径 | 大小 | 重建 |
|---|---|---|
| `.venv/` | 0.85 GiB | `venv` + `deps` 步 |
| `qlora_lab/.venv/` | 0.84 GiB | `qlora` 步（可选） |
| `qlora_lab/model/` | 1.65 GiB | `assets` 步：从 `hf-mirror.com` 下载 |
| `cpu_lab/checkpoints/` | 4.13 GiB | `dcp` 步：`convert_qwen35_hf.py` |
| `cpu_lab/hf/`、`qlora_lab/hf/` | ~1 GiB | 下载缓存 |
| `cpu_lab/logs*/` | — | 运行时产物 |

> 不用 Git LFS：免费额度 10 GiB 存储 / 10 GiB 月流量，权重就 6.6 GiB，
> 而且上传比从镜像站重下还慢。

---

## 四、新机器上手（Windows 11）

### 4.0 前置检查

```powershell
# 1. 确认 Python 3.11（torchtitan pinned 版本）
python --version          # 期望 3.11.x

# 2. 确认有 uv（没有的话 bootstrap 会退回 python -m venv，也能用）
where.exe uv

# 3. 确认盘符空间 —— 仓库需要 ~7 GB
Get-PSDrive -PSProvider FileSystem | Select-Object Name,@{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}
```

> 本机工具的已知坑：Bash 工具的 PATH 可能损坏；PowerShell 的 stdout 不回传。
> 需要看命令输出时，**把结果写进文件再用 Read 读**。

### 4.1 一键重建（推荐）

```bash
git clone https://github.com/bantianyue/torchtitan.git
cd torchtitan
python cpu_lab/bootstrap.py
```

6 个步骤，全部幂等：

| 步 | 做什么 | 产出 |
|---|---|---|
| `venv` | 建 CPython 3.11 的 `.venv` | `.venv/` |
| `deps` | **先从 PyTorch CPU 索引装 torch**，再从 PyPI 装其余包 | 环境就绪 |
| `assets` | 从 `hf-mirror.com` 抓 `Qwen/Qwen3.5-0.8B` | `qlora_lab/model/` |
| `dcp` | HF safetensors → torchtitan DCP 布局 | `cpu_lab/checkpoints/qwen35_0_8b_hf/` |
| `data` | 生成植入语料 + **单进程**预热数据集缓存 | `cpu_lab/planted/`、HF 缓存 |
| `smoke` | 单 rank 跑 2 步（随机初始化） | `cpu_lab/logs_qwen35/smoke/` |
| `qlora` | *（可选，默认不跑）* 建 `qlora_lab/.venv` | 第二条轨的环境 |

> `deps` 分两阶段是必须的：`requirements-cpu.txt` 钉了 `torch==2.12.1`，
> 直接从 PyPI 装在 Windows 上会拉到 ~2.5 GB 的 **CUDA** 构建。

失败时脚本会打印**从失败步继续**的命令：

```bash
python cpu_lab/bootstrap.py --steps dcp,data,smoke   # 只跑这几步
python cpu_lab/bootstrap.py --skip assets            # 权重已就位时跳过下载
python cpu_lab/bootstrap.py --steps qlora            # 只建 qlora_lab 环境
```

### 4.2 手动版

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.12.1 torchdata==0.11.0
uv pip install --python .venv/Scripts/python.exe -r cpu_lab/requirements-cpu.txt
uv pip install --python .venv/Scripts/python.exe -e .

python cpu_lab/convert_qwen35_hf.py
python cpu_lab/prewarm_dataset.py --hf-cache cpu_lab/hf
```

---

## 五、验证方法

分六层，**逐层通过再往上走**。每层的「通过判据」是可机器核对的具体值，不是感觉。

### L0 — 环境与补丁链

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

- 判据：输出 `2.12.1`（或带 `+cpu`），`False`
- 失败处理：若出现 CUDA 构建，说明 `deps` 阶段没走 CPU 索引，删 `.venv` 重跑 `venv,deps`

### L1 — 配置干跑（不跑真通信）

```bash
.venv/Scripts/python.exe -u cpu_lab/run_cpu_train.py --combo qwen35 -n 1 \
    --module cpu_lab.qwen35 --config qwen35_0_8b_smoke \
    --dry-run --no-warmup --log-dir cpu_lab/logs_qwen35/_verify
```

- 判据：正常退出，`RUN SUMMARY` 打印出 `module/config = cpu_lab.qwen35 / qwen35_0_8b_smoke`
- 本机实测：**13.6 s**
- 失败处理：若报「找不到配置函数」，十有八九是 `cpu_lab_child.py` 的模块门控拼写问题（见坑 3）

### L2 — Smoke：单 rank 随机初始化 2 步

```bash
.venv/Scripts/python.exe -u cpu_lab/run_cpu_train.py --combo qwen35 -n 1 \
    --module cpu_lab.qwen35 --config qwen35_0_8b_smoke \
    --hf-cache cpu_lab/hf --no-warmup --show-logs
```

- 判据：`steps=2`，`last_step=2`，`has_real_loss=True`，loss 在 **12.x** 量级（随机初始化）
- 失败处理：若崩在 `torch.compile` / triton，说明 `cpu_compat` 没在导入 torchtitan 前生效

### L3 — 真实权重 SFT：单 rank 10 步

```bash
.venv/Scripts/python.exe -u cpu_lab/run_cpu_train.py --combo qwen35 -n 1 \
    --threads 4 --module cpu_lab.qwen35 --config qwen35_0_8b_sft_bf16 \
    --hf-cache cpu_lab/hf --no-warmup --keep-outputs \
    --log-dir cpu_lab/logs_qwen35/sft_bf16_1r --show-logs
```

- **判据（最关键的一条）：step 1 的 loss ≈ 1.3。**
  随机初始化是 12.8；第一步就掉到 1.3 才说明 HF→DCP 转换和 weight tying 都正确。
  若 step 1 仍是 12.x，权重根本没加载上。
- 参考值：step 10 loss 0.24605，grad_norm 145 → 41.25，wall 819.1 s，峰值 RSS 7.69 GiB

### L4 — 并行正确性

判据取决于维度，**别搞反**：

- **TP 不对 batch 分片** ⇒ 所有 rank 的 loss 必须**逐位相同**。不一致说明漏了集合通信。
- **FSDP 会分片 batch** ⇒ 跨策略比较 loss 前必须先对齐 global batch。
- **grad_norm 全 rank 相同**（PP 下也一样），因为裁剪是同步的。

自动化核对：`summary.txt` 里的 `distinct loss values` 与 `distinct grad_norms` 两行。

### L5 — 推理验证（probe）

```bash
.venv/Scripts/python.exe cpu_lab/run_cpu_train.py --combo <preset> -n 8 \
    --collectives --probe --dataset-path cpu_lab/planted --steps N
```

- 判据：`probe.json` 里模型能预测出植入的暗号
- **两个会让结论假阴性的坑**（都会误判成「权重没训好」）：
  1. 不要无条件把 `lm_head` 重绑到 embedding。tied flavor 下无害，但 PP 用的 **untied**
     flavor 下会把训好的输出头换成 embedding，输出变噪声。
  2. prompt 必须带 **BOS**，并与训练时的 position 对齐。

### 报告

`--report`（默认开）会渲染 `cpu_lab/reports/<combo>.html`，自包含无 CDN，可离线打开。
用已有日志重出报告而不重训：

```bash
python cpu_lab/report.py --log-dir D --combo C
```

---

## 六、后续要怎么做

### 6.1 立刻：重建 + 验收

```bash
git clone https://github.com/bantianyue/torchtitan.git
cd torchtitan
python cpu_lab/bootstrap.py          # 自动跑完 L0–L2
```

然后照第五节把 L0–L3 逐条对一遍。

### 6.2 短期：重跑 SFT 并留住产物

先记录新机器的内存基线，再决定 `--threads`：

```bash
python cpu_lab/run_cpu_train.py --combo qwen35 -n 1 --threads 4 \
    --module cpu_lab.qwen35 --config qwen35_0_8b_sft_bf16 \
    --hf-cache cpu_lab/hf --no-warmup --keep-outputs \
    --log-dir cpu_lab/logs_qwen35/sft_bf16_1r --show-logs
```

跑完**立刻**把 checkpoint 挪出会被清空的目录：

```bash
mkdir -p cpu_lab/checkpoints/sft-<date>
mv cpu_lab/outputs/qwen35/checkpoint/step-10 cpu_lab/checkpoints/sft-<date>/
```

（`cpu_lab/checkpoints/` 已被 gitignore，但不会被启动器清空。）

### 6.3 中期：在新机器上重新测量规模上限

旧机器的数字**只是参考**，新机器配置不同必须重测：

```bash
python cpu_lab/scaling.py --log-dir cpu_lab/logs_scale/<flavor>   # >200M 加 --no-warmup
python cpu_lab/scaling.py                                          # → reports/scaling.html
```

旧机器拟合出的公式（换机器后可直接套用）：

```
commit(8 rank 合计) = 41.2 B/param × N + 4.08 GiB
RSS                 = 20.3 B/param × N + 3.35 GiB
每步耗时             = 0.170 s/M + 1.1 s
```

关键观察：**截距（每 rank 固定 ~0.5 GiB）主导**，所以减 rank 数比缩模型更能抬高天花板。

### 6.4 多 rank 路线图（按这个顺序试，别跳步）

1. **2 rank** + `qwen35_0_8b_bf16_multi`（已放宽超时到 1800 s）→ 应能过
2. **4 rank** → 旧机器上是 `access violation`，新机器若内存更大可能过
3. **8 rank** → **必须带 PP/TP 维度，不能只用 `dp_shard=8`**

原因（内存模型）：

```
总 commit ≈ 模型状态常量 + 每 rank 不可分片开销 × rank 数

bf16 模型状态常量 = 6.83 GiB（参数 1.71 + 梯度 1.71 + Adam 3.41）
纯 FSDP 每 rank 开销 ≈ 4.8 GiB
    ├─ all-gather 输出缓冲   1.71 GiB
    ├─ 展平梯度缓冲           1.71 GiB
    └─ Python + torch 运行时  ~1.4 GiB
```

`dp_shard` **不缩小**那两项缓冲 ⇒ 纯 `dp_shard=8` ≈ 45 GiB，32 GiB 机器上仍贴顶。
而 **PP/TP 会真正缩小** ⇒ `pp2×tp2×dp_shard2`（8 rank）≈ **25 GiB**，32 GiB 机器可行。

**并行策略可用性矩阵（旧机器实测）**

| 策略 | 状态 |
|---|---|
| `single` / `fsdp` / `tp` / `tp_fsdp` / `tp_sp` / `pp` / `pp_tp` / `pp_fsdp` | ✅ |
| `pp_tp_fsdp` | ⚠️ 必须 `--sp false` |
| `hsdp` | ⚠️ 4 rank（2×2）可用；8 rank（2×4）挂 |
| `cp` / `tp_cp` | ❌ CPU SDPA 派发到 `_scaled_dot_product_flash_attention_for_cpu`，无 DTensor 分片策略 |

### 6.5 关于 4bit 量化（原始需求）

**结论：在 CPU 上不要做 4bit，已实测放弃。**

- torchtitan 本身**没有**量化训练能力（无 bitsandbytes / QLoRA 路径）。
- 旁证：独立目录 `qlora_lab/` 用 HF + PEFT 实测过。CPU 上 bitsandbytes 只能走回退内核，
  每次矩阵乘前先反量化成 bf16 ⇒ **比原生 fp32 更慢**。
- 而 0.8B 主干 fp32 也只有 ~3.4 GB，省内存靠 **LoRA 冻结主干**（不产生梯度、不占优化器状态），
  不靠量化。

`qlora_lab/` 的实测（4 核 CPU）：13.3 s/step、12.6 token/s、峰值 RSS 3.15 GiB，
LoRA r=8 可训练 5,111,808 参数（0.599%）。**瓶颈是时间不是内存**。

---

## 七、已知坑

### 1. 多 rank 崩溃的真因是集合通信超时，不是内存

`Trainer.Config.comm` 默认 **init 300 s、之后收紧到 100 s** —— 这是按加速器定的。
但分页的 CPU rank 一次 all-gather 合法地要几分钟，「首步后收紧到 100 s」等于保证每步都超时。

```python
CommConfig(init_timeout_seconds=1800, train_timeout_seconds=1800)
# 或 CLI: --comm.init_timeout_seconds 1800 --comm.train_timeout_seconds 1800
```

已内置配置 `qwen35_0_8b_bf16_multi`。**这是诊断手段不是生产设置**：
放宽后能跑完 ⇒ 「装得下但在换页」；1800 s 仍超时 ⇒ 真死锁。

判断依据：崩溃时 commit 还余 **2.08 GiB**（2 rank），说明不是内存耗尽。

### 2. 数据集 shuffle 缓存路径与 rank 无关

`shuffle(seed=42)` 在所有 rank 上产生**同一个** permutation hash ⇒ 8 个 rank 抢写同一个
`cache-*.arrow`，rank 1 报 `OSError: [Errno 22]`，其余 rank 卡在数据集准备里等。

解法：先跑 `prewarm_dataset.py`（单进程、不建模型）建好缓存，再 `--no-warmup` 起多 rank。
**不是 MAX_PATH 问题**（路径 173 字符、`LongPathsEnabled=1`）。

### 3. 模块门控拼写不会报错

`cpu_lab_child.py` 里判 `"qwen3_5" in module`，而模块名是 `cpu_lab.qwen35`（无下划线）
⇒ 补丁没装 ⇒ `config_registry` 导入 `fla` 失败 ⇒ `ConfigManager` 的
`except ImportError: continue` **静默吞掉** ⇒ 最后只报「找不到配置函数」。

只读 `sys.argv` 的门控要同时匹配两种拼写。

### 4. `training.dtype` ≠ `mixed_precision_param`

前者在 meta 构建期 `torch.set_default_dtype`，使参数 / 梯度 / Adam **全部**为该 dtype；
后者只改前向 / 通信 dtype。**要省内存必须用前者**（fp32 13.65 GB → bf16 6.83 GB）。

### 5. 必须开 weight tying

HF `tie_word_embeddings=true` ⇒ 磁盘上**没有** `lm_head.weight` ⇒ 不开时 `dcp.load` 直接报
`Missing key in checkpoint state_dict: lm_head.weight`。
（源码里「torchtitan 不支持 tied embedding」的注释**已过时**。）

### 6. 无效手段（别再试）

`PYTORCH_CPU_ALLOC_CONF=expandable_segments:True` 在 torch 2.12.1 上
**被静默接受但无效**（该选项仅 CUDA）。分配 churn A/B 实测无差异。

### 7. Windows 专属

- torch CPU 构建无 libuv ⇒ `TCPStore` 必须显式 `use_libuv=False`
  ⇒ **torchrun 不可用**，要自己 spawn 子进程设 `RANK/LOCAL_RANK/WORLD_SIZE`。
- `initial_load_path` 校验是 POSIX-only（`startswith("/")`），拒 `D:\...`。
- 新机器仍是 Windows 11 ⇒ 这两条补丁**仍然需要**。

---

## 八、机器档案（旧机器，仅供对比）

| 项 | 旧机器 |
|---|---|
| CPU | i7-1165G7，4 核 / 8 线程（逻辑 CPU 8），无 GPU |
| 内存 | 物理 16 GiB（15.73 GiB 可用） |
| commit 上限 | 45.63 GiB；桌面基线常驻 25–30 GiB |
| torch | 2.12.1+cpu，仅 gloo，无 triton |
| 单 rank bf16 | 64 s/step，峰值 RSS 7.69 GiB |
| 2 rank | ~140 s/step（**比 1 rank 慢 2.2 倍**） |

**换 32 GiB 机器的结论**：8 rank 的 `pp2×tp2×dp_shard2` **能跑通**（~25 GiB），但**不会更快**。
瓶颈是 gloo over TCP 每次 all-gather 搬 1.71 GiB，**多开 rank 增加的是通信不是算力**。
32 GiB 上的合理配置是 **4 rank × 2 线程**。投资顺序：**更多物理核 > 更多内存 > 更多 rank**。

---

## 九、常用命令速查

```bash
# 并行策略一览
python cpu_lab/run_cpu_train.py --list

# 缩放阶梯（TinyGPT，找当前机器的模型规模上限）
python cpu_lab/scaling.py --log-dir cpu_lab/logs_scale/<flavor>   # >200M 加 --no-warmup

# Qwen3.5 单 rank 真实权重 SFT（注意 --keep-outputs）
python cpu_lab/run_cpu_train.py --combo qwen35 -n 1 --threads 4 \
    --module cpu_lab.qwen35 --config qwen35_0_8b_sft_bf16 \
    --hf-cache cpu_lab/hf --no-warmup --keep-outputs

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
`--keep-outputs` 防止启动器清空 `cpu_lab/outputs/<combo>/`。
