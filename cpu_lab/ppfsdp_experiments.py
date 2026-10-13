#!/usr/bin/env python
"""Two experiments that separate 'PPxFSDP is broken' from 'PPxFSDP does not fit'.

Experiment 1 (control): the same pp_fsdp combination on TinyGPT, where memory is
not a factor. If it completes, the PP x FSDP code path itself is sound and the
Qwen2.5 failure is about size, not logic.

Experiment 2 (reduced footprint): the same Qwen2.5 pp_fsdp run with fewer
microbatches and one OMP thread per rank, i.e. the cheapest way to cut the job's
peak. If it completes, the wall was memory and this is the recipe.

Results are appended to results.txt as each experiment finishes, so a partial run
still tells you something.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
OUT = os.path.join(HERE, "ppfsdp_experiments.txt")


def run(label: str, cmd: list[str]) -> None:
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(f"\n{'=' * 78}\n{label}\ncmd: {' '.join(cmd)}\n{'=' * 78}\n")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, errors="replace")
    dt = time.time() - t0
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(f"exit={proc.returncode}  wall={dt:.1f}s\n")
        tail = (proc.stdout or "")[-3000:]
        fh.write(tail)
        if proc.stderr:
            fh.write("\n--- stderr ---\n" + proc.stderr[-2000:])
    print(f"[experiments] {label}: exit={proc.returncode} wall={dt:.1f}s", flush=True)


def main() -> None:
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("PP x FSDP experiments on Qwen2.5-0.5B / TinyGPT\n")

    # 1. Control: same combination, tiny model, memory is a non-issue.
    run(
        "experiment 1: TinyGPT pp_fsdp x4 (control)",
        [
            PY, os.path.join("cpu_lab", "run_cpu_train.py"),
            "--combo", "pp_fsdp", "-n", "4",
            "--steps", "2", "--threads", "2",
            "--log-dir", os.path.join("cpu_lab", "logs_qwen25", "exp1_tinygpt"),
            "--no-warmup",
        ],
    )

    # 2. Qwen2.5 pp_fsdp with the footprint cut: 2 microbatches, 1 thread/rank.
    run(
        "experiment 2: Qwen2.5 pp_fsdp x4, local_batch_size=2, threads=1",
        [
            PY, os.path.join("cpu_lab", "run_cpu_train.py"),
            "--combo", "pp_fsdp", "-n", "4",
            "--module", "cpu_lab.qwen25", "--config", "qwen25_0_5b_pp_bf16",
            "--steps", "2", "--threads", "1",
            "--local-batch-size", "2",
            "--stagger", "5",
            "--log-dir", os.path.join("cpu_lab", "logs_qwen25", "exp2_qwen25_small"),
            "--no-warmup",
        ],
    )

    # 3. Same as 2 but with the microbatch count cut to 1 as well, the smallest
    #    footprint this config can express.
    run(
        "experiment 3: Qwen2.5 pp_fsdp x4, local_batch_size=1, threads=1",
        [
            PY, os.path.join("cpu_lab", "run_cpu_train.py"),
            "--combo", "pp_fsdp", "-n", "4",
            "--module", "cpu_lab.qwen25", "--config", "qwen25_0_5b_pp_bf16",
            "--steps", "2", "--threads", "1",
            "--local-batch-size", "1",
            "--stagger", "5",
            "--log-dir", os.path.join("cpu_lab", "logs_qwen25", "exp3_qwen25_min"),
            "--no-warmup",
        ],
    )
    print("[experiments] all done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
