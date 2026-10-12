#!/usr/bin/env bash
# qlora_lab environment bootstrap.
#   - creates an ISOLATED venv (never touches torchtitan/.venv used by cpu_lab)
#   - installs the Qwen3.5 QLoRA stack
#   - prefetches the Qwen3.5-0.8B weights into D: (C: only has ~13 GiB free)
#
# usage:  bash qlora_lab/setup_env.sh 2>&1 | tee qlora_lab/setup.log
set -euo pipefail

ROOT=/d/02_source/torchtitan
LAB="$ROOT/qlora_lab"
UV=/c/Users/twfehh7/.local/bin/uv.exe

# EVERYTHING lands on D:. By default uv caches wheels under
# %LOCALAPPDATA%\uv\cache on C: -- that wrote ~500 MB to a disk with only
# ~13 GiB free. Pin the cache, the HF cache and the model next to this script.
export UV_CACHE_DIR="$LAB/.uvcache"
export UV_LINK_MODE=copy          # venv and cache share a filesystem now, but
                                  # keep it explicit: D: vs C: differs from the
                                  # uv-managed interpreter's location
export HF_HOME="$LAB/hf"
export HF_HUB_DISABLE_XET=1       # the Xet backend stalls on this box

step() { echo; echo "===== $* ====="; }

mkdir -p "$LAB"

step "1/4 create isolated venv (uv-managed CPython 3.11)"
"$UV" venv "$LAB/.venv" --python 3.11
PY="$LAB/.venv/Scripts/python.exe"
"$PY" -c "import sys; print(sys.version); print(sys.executable)"

step "2/4 install torch (CPU wheel index)"
"$UV" pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cpu

step "3/4 install transformers>=5.2.0 + peft + accelerate + bitsandbytes + datasets"
"$UV" pip install --python "$PY" \
  "transformers>=5.2.0" peft accelerate bitsandbytes datasets safetensors

step "4/4 verify + report versions"
"$PY" - <<'PYEOF'
import importlib, json, platform, torch
mods = ["torch","transformers","peft","accelerate","bitsandbytes","datasets","safetensors","tokenizers","huggingface_hub"]
out = {}
for m in mods:
    try:
        out[m] = getattr(importlib.import_module(m), "__version__", "?")
    except Exception as e:
        out[m] = f"MISSING ({type(e).__name__})"
print(json.dumps(out, indent=1))
print("cuda:", torch.cuda.is_available(), "| cpu capability:", torch.backends.cpu.get_cpu_capability())
print("threads:", torch.get_num_threads())
try:
    import bitsandbytes as bnb
    print("bnb cuda available:", bnb.__dict__.get("cuda_spec", None) is not None)
except Exception as e:
    print("bnb import problem:", e)
PYEOF

step "prefetch Qwen/Qwen3.5-0.8B weights -> $HF_HOME"
"$PY" - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3.5-0.8B", allow_patterns=[
    "*.json","*.safetensors","*.txt","*.py","*.model","tokenizer*","*.tiktoken"
])
print("snapshot:", p)
tot = sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(p) for f in fs)
print("size: %.2f GiB" % (tot/1024**3))
PYEOF

step "DONE"
