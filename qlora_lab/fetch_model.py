#!/usr/bin/env python
"""Fetch the Qwen3.5-0.8B snapshot with the classic HTTP downloader.

hf-xet stalls on this box (the blob froze at ~536 MB for minutes with the
process still alive), so disable Xet explicitly and retry with backoff.
Resumes from the existing *.incomplete blob.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-0.8B")

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HOME", os.path.join(HERE, "hf"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download  # noqa: E402


def dir_size(path):
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot


def main():
    print(f"HF_HOME={os.environ['HF_HOME']}  XET disabled", flush=True)
    last = None
    for attempt in range(1, 7):
        try:
            t0 = time.time()
            p = snapshot_download(REPO, max_workers=4, etag_timeout=30)
            sz = dir_size(p)
            print(f"\nOK attempt {attempt}: {p}", flush=True)
            print(f"size: {sz/1024**3:.2f} GiB in {time.time()-t0:.0f}s", flush=True)
            return 0
        except Exception as e:
            sz = dir_size(os.path.join(os.environ["HF_HOME"], "hub"))
            print(f"\nattempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
            print(f"  cache now {sz/1024**3:.2f} GiB", flush=True)
            time.sleep(min(5 * attempt, 20))
    print("FAILED after retries", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
