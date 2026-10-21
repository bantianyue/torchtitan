#!/usr/bin/env python3
"""Rebuild the CPU training lab from a fresh clone.

Why this exists
---------------
The repository is committed **without** anything that can be rebuilt:

    .venv/                   venvs bake in absolute paths, useless elsewhere
    qlora_lab/.venv/
    qlora_lab/model/         HF weights + tokenizer, ~1.65 GiB
    qlora_lab/hf/            hub cache, ~0.92 GiB (incl. download fragments)
    cpu_lab/checkpoints/     HF weights converted to DCP, ~4.13 GiB
    cpu_lab/logs*/           per-rank .log files

Cloning therefore costs ~150 MB instead of ~7 GB, and nothing in the repo is
a stale copy of something downloadable. This script puts all of it back.

What it does
------------
    venv     create .venv with CPython 3.11
    deps     install the pinned CPU torch build, then the rest from PyPI
    assets   download Qwen/Qwen3.5-0.8B into qlora_lab/model/
    dcp      convert those HF weights into torchtitan's DCP layout
    data     build the planted corpus and pre-warm the dataset cache
    smoke    run a 2-step single-rank training to prove the stack works
    qlora    OPTIONAL (off by default) build qlora_lab/.venv, the separate
             environment used by the HF/PEFT fine-tuning track

Steps other than ``qlora`` run by default; pass ``--steps qlora`` to build
only that environment.

Every step is idempotent and each can be skipped, so a partial failure does not
force starting over::

    python cpu_lab/bootstrap.py                       # everything
    python cpu_lab/bootstrap.py --steps deps,dcp      # just these
    python cpu_lab/bootstrap.py --skip assets,dcp     # weights already present

Platform notes
--------------
Windows and Linux differ in three ways that matter here:

- venv layout: ``Scripts/python.exe`` vs ``bin/python``
- torch build: Windows CPU wheels ship without libuv, which is why
  ``cpu_compat`` has to force ``use_libuv=False`` on TCPStore. Linux does not
  need that patch.
- checkpoint paths: torchtitan validates ``initial_load_path`` with
  ``startswith("/")``, which rejects ``D:\\...``. ``cpu_compat`` patches it to
  ``os.path.isabs``. Linux paths were never affected.

The patches themselves are platform-gated inside ``cpu_compat``, so nothing
here needs to branch.

The hard constraint preserved from the original setup: **nothing is written to
the system drive.** HF caches, uv caches and outputs all live under the repo
root, so cloning the repo to a drive with free space is sufficient.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

VENV = REPO_ROOT / ".venv"
VENV_PY = (
    VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
)

# Everything stays inside the working tree so the system drive is never touched
# and a whole experiment can be deleted by removing one directory.
HF_CACHE = REPO_ROOT / "cpu_lab" / "hf"
UV_CACHE = REPO_ROOT / ".uv_cache"
HF_MODEL_DIR = REPO_ROOT / "qlora_lab" / "model"
DCP_DIR = REPO_ROOT / "cpu_lab" / "checkpoints" / "qwen35_0_8b_hf"

MODEL_REPO = "Qwen/Qwen3.5-0.8B"
REQUIREMENTS = HERE / "requirements-cpu.txt"

# Straight to the mirror: huggingface.co was not reachable from the network
# this was developed on. Override with --hf-endpoint if the new machine can
# reach the origin directly.
# Xet is disabled because its backend wedges mid-transfer here.
HF_ENDPOINT = "https://hf-mirror.com"

# qlora is opt-in: the torchtitan track does not need it, and it pulls a
# second, much newer torch (2.14.1) that must never land in the main .venv.
CORE_STEPS = ("venv", "deps", "assets", "dcp", "data", "smoke")
OPTIONAL_STEPS = ("qlora",)
ALL_STEPS = CORE_STEPS + OPTIONAL_STEPS


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT, env: dict[str, str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(cwd), env=env)


def _resolve_uv() -> str | None:
    """Prefer the real uv binary; fall back to whatever pip the interpreter has."""
    found = shutil.which("uv")
    if found:
        return found
    for candidate in (
        Path.home() / ".local" / "bin" / ("uv.exe" if os.name == "nt" else "uv"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def base_env(endpoint: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        HF_ENDPOINT=endpoint or HF_ENDPOINT,
        HF_HOME=str(HF_CACHE),
        HF_DATASETS_CACHE=str(HF_CACHE / "datasets"),
        HF_HUB_DISABLE_XET="1",
        UV_CACHE_DIR=str(UV_CACHE),
        PYTHONUNBUFFERED="1",
        # Windows CPU torch has no libuv; cpu_compat also forces this, but set it
        # here so any helper process inherits the right default.
        USE_LIBUV="0",
    )
    return env


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #

def step_venv(env: dict[str, str], args: argparse.Namespace) -> int:
    if VENV_PY.exists():
        print(f"[bootstrap] reusing existing venv at {VENV}")
        return 0

    uv = _resolve_uv()
    if uv:
        cmd = [uv, "venv", str(VENV), "--python", "3.11"]
    else:
        print("[bootstrap] uv not found; falling back to python -m venv")
        cmd = [sys.executable, "-m", "venv", str(VENV)]

    HF_CACHE.mkdir(parents=True, exist_ok=True)
    UV_CACHE.mkdir(parents=True, exist_ok=True)
    return _run(cmd, env=env)


def step_deps(env: dict[str, str], args: argparse.Namespace) -> int:
    """Install torch from the CPU index, then everything else from PyPI.

    The split is the whole point: ``requirements-cpu.txt`` pins ``torch==2.12.1``,
    and installing that straight off PyPI on Windows pulls the ~2.5 GB CUDA
    build. The CPU index has to win for torch, while PyPI still has to serve
    the other 77 packages.
    """
    python = str(VENV_PY)
    uv = _resolve_uv()

    # uv resolves an entire environment at once, so it can honour a per-package
    # index; plain pip cannot, hence the two-phase fallback below.
    if uv:
        rc = _run([uv, "pip", "install", "--python", python,
                   "--index-url", "https://download.pytorch.org/whl/cpu",
                   "torch==2.12.1", "torchdata==0.11.0"], env=env)
        if rc != 0:
            return rc
    else:
        rc = _run([python, "-m", "pip", "install",
                   "--index-url", "https://download.pytorch.org/whl/cpu",
                   "torch==2.12.1", "torchdata==0.11.0"], env=env)
        if rc != 0:
            return rc

    # Second pass: the remaining pins, minus torch, from PyPI.
    stripped = HERE / "_requirements_cpu_notorch.txt"
    lines = [
        ln for ln in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lower().startswith(("torch==", "torchdata=="))
    ]
    stripped.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        if uv:
            rc = _run([uv, "pip", "install", "--python", python,
                       "-r", str(stripped)], env=env)
        else:
            rc = _run([python, "-m", "pip", "install",
                       "-r", str(stripped)], env=env)
    finally:
        stripped.unlink(missing_ok=True)

    if rc != 0:
        return rc

    # torchtitan is used from this tree, not from a wheel: the CPU patches are
    # applied at runtime by cpu_lab/ and the package is imported in place.
    if uv:
        rc = _run([uv, "pip", "install", "--python", python,
                   "-e", str(REPO_ROOT)], env=env)
    else:
        rc = _run([python, "-m", "pip", "install",
                   "-e", str(REPO_ROOT)], env=env)
    return rc


def step_assets(env: dict[str, str], args: argparse.Namespace) -> int:
    """Download the HF snapshot (weights + tokenizer) into qlora_lab/model/."""
    HF_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    script = (
        "import os,sys\n"
        "from huggingface_hub import snapshot_download\n"
        "p=snapshot_download("
        f"repo_id={MODEL_REPO!r}, local_dir={str(HF_MODEL_DIR)!r}, "
        "max_workers=8)\n"
        "print('[assets]', p)\n"
    )
    return _run([str(VENV_PY), "-u", "-c", script], env=env)


def step_dcp(env: dict[str, str], args: argparse.Namespace) -> int:
    """HF safetensors -> torchtitan DCP (this is what ``initial_load_path`` reads)."""
    return _run([str(VENV_PY), "-u", str(HERE / "convert_qwen35_hf.py"),
                 "--input-dir", str(HF_MODEL_DIR),
                 "--output-dir", str(DCP_DIR)], env=env)


def step_data(env: dict[str, str], args: argparse.Namespace) -> int:
    """Planted corpus + the single-process dataset-cache warmup.

    The warmup is not optional at >1 rank: ``shuffle(seed=42)`` hashes to the
    same permutation on every rank, so all ranks try to write the *same*
    ``cache-*.arrow`` and rank 1 dies with ``OSError: [Errno 22]``.
    """
    rc = _run([str(VENV_PY), "-u", str(HERE / "planted_corpus.py")], env=env)
    if rc != 0:
        return rc
    return _run([str(VENV_PY), "-u", str(HERE / "prewarm_dataset.py"),
                 "--hf-cache", str(HF_CACHE)], env=env)


def step_smoke(env: dict[str, str], args: argparse.Namespace) -> int:
    threads = args.smoke_threads
    return _run([str(VENV_PY), "-u", str(HERE / "run_cpu_train.py"),
                 "--combo", "qwen35",
                 "-n", "1",
                 "--threads", str(threads),
                 "--module", "cpu_lab.qwen35",
                 "--config", "qwen35_0_8b_smoke",
                 "--hf-cache", str(HF_CACHE),
                 "--log-dir", str(REPO_ROOT / "cpu_lab" / "logs_qwen35" / "smoke"),
                 "--show-logs"], env=env)


def step_qlora(env: dict[str, str], args: argparse.Namespace) -> int:
    """Build ``qlora_lab/.venv`` -- the HF/PEFT track's own environment.

    Why a second venv at all: Qwen3.5 needs ``transformers >= 5.2.0``, which
    in turn wants a torch newer than the 2.12.1 the torchtitan track is pinned
    to. Installing both into one environment silently breaks one of them, so
    the two tracks stay separate and ``qlora_lab/`` never imports
    ``cpu_lab``'s venv.

    Also replaces ``qlora_lab/setup_env.sh``: that is a bash script, and the
    target machine is Windows 11, where bash is optional.
    """
    lab = REPO_ROOT / "qlora_lab"
    venv = lab / ".venv"
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    uv = _resolve_uv()

    if py.exists():
        print(f"[bootstrap] qlora venv already present at {venv}")
    else:
        venv.parent.mkdir(parents=True, exist_ok=True)
        if uv:
            rc = _run([uv, "venv", str(venv), "--python", "3.11"], env=env)
        else:
            rc = _run([sys.executable, "-m", "venv", str(venv)], env=env)
        if rc != 0:
            return rc

    # One install pass. The pins in qlora_lab/requirements.txt already carry the
    # +cpu local version (torch==2.14.1+cpu), and the CPU index serves it, so
    # unlike the main .venv there is no need for the two-phase split.
    if uv:
        return _run([uv, "pip", "install", "--python", str(py),
                     "--index-url", "https://download.pytorch.org/whl/cpu",
                     "-r", str(lab / "requirements.txt")], env=env)
    return _run([str(py), "-m", "pip", "install",
                 "--index-url", "https://download.pytorch.org/whl/cpu",
                 "-r", str(lab / "requirements.txt")], env=env)


STEPS = {
    "venv": step_venv,
    "deps": step_deps,
    "assets": step_assets,
    "dcp": step_dcp,
    "data": step_data,
    "smoke": step_smoke,
    "qlora": step_qlora,
}


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--steps",
        default=",".join(CORE_STEPS),
        help=f"comma-separated subset of: {','.join(ALL_STEPS)} "
        f"(default: {','.join(CORE_STEPS)})",
    )
    p.add_argument(
        "--skip",
        default="",
        help="comma-separated steps to skip, e.g. --skip assets,dcp",
    )
    p.add_argument(
        "--smoke-threads",
        type=int,
        default=max(1, (os.cpu_count() or 4)),
        help="OMP threads for the smoke run (default: all cores)",
    )
    p.add_argument(
        "--hf-endpoint",
        default=HF_ENDPOINT,
        help=f"HF hub endpoint for the assets step (default: {HF_ENDPOINT}). "
        "Use https://huggingface.co if the new network reaches the origin.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    wanted = {s.strip() for s in args.steps.split(",") if s.strip()}
    skipped = {s.strip() for s in args.skip.split(",") if s.strip()}
    unknown = (wanted | skipped) - set(ALL_STEPS)
    if unknown:
        raise SystemExit(
            f"unknown step(s): {', '.join(sorted(unknown))}; known: {', '.join(ALL_STEPS)}"
        )
    plan = [s for s in ALL_STEPS if s in wanted and s not in skipped]
    if not plan:
        raise SystemExit("nothing to do")

    print(f"[bootstrap] repo root : {REPO_ROOT}")
    print(f"[bootstrap] HF cache  : {HF_CACHE}")
    print(f"[bootstrap] uv cache  : {UV_CACHE}")
    print(f"[bootstrap] plan      : {' -> '.join(plan)}")

    env = base_env(args.hf_endpoint)
    for name in plan:
        print(f"\n{'=' * 66}\n[bootstrap] step: {name}\n{'=' * 66}", flush=True)
        rc = STEPS[name](env, args)
        if rc != 0:
            print(f"\n[bootstrap] FAILED at step {name!r} (exit {rc})", file=sys.stderr)
            print(
                f"[bootstrap] fix the cause, then resume with:\n"
                f"    python cpu_lab/bootstrap.py --steps "
                f"{','.join(plan[plan.index(name):])}",
                file=sys.stderr,
            )
            return rc

    print("\n[bootstrap] done. Next:")
    print("    python cpu_lab/run_cpu_train.py --combo qwen35 -n 1 \\")
    print(f"        --module cpu_lab.qwen35 --config qwen35_0_8b_sft_bf16 \\")
    print(f"        --hf-cache {HF_CACHE.relative_to(REPO_ROOT)}")
    print("\nSee cpu_lab/MIGRATION.md for the full recipe and measured limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
