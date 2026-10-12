# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate a corpus with a *planted* secret, to prove training actually ran.

A loss curve going down is weak evidence: it can come from learning byte
frequencies and nothing else. The strong test is **memorisation of content that
cannot be guessed** -- a string that exists only in this corpus. So this module
writes a dataset in which two made-up launch codes appear, each anchored to its
own fixed prefix, and records what the model should say when handed those
prefixes. ``probe_inference.py`` then measures whether the trained weights
predict them and whether random-init weights do not.

The tokenizer used by this config is byte-level (``tests/assets/tokenizer``,
vocab 2020), so the text round-trips as characters. Two things follow:

- Whatever is written here becomes, after tokenisation, a run of per-character
  ids -- no subword merging to reason about.
- Spaces are dropped by this tokenizer, so the prompt is compared **as ids**,
  taken from the real tokenizer, rather than by string matching. Nothing here
  assumes what the tokeniser does with whitespace.

The corpus is deliberately skewed: two planted sentences repeated many times,
diluted with a little real C4 text so the model is not simply fitting one
constant string. That is the standard "overfit a tiny dataset" setup, and it is
what makes a memorisation result readable in a few dozen steps.
"""

from __future__ import annotations

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
# The dataset *directory* must contain nothing but data files: HuggingFace's
# ``load_dataset(<dir>)`` globs everything under it and infers one schema from
# all of them, so a manifest.json left here makes the load fail with
# "column names don't match". Hence the manifest lives beside the directory.
OUT_DIR = os.path.join(HERE, "planted")
MANIFEST_PATH = os.path.join(HERE, "probe_manifest.json")
C4_TEST = os.path.join(REPO_ROOT, "tests", "assets", "c4_test", "data.json")
TOKENIZER_DIR = os.path.join(REPO_ROOT, "tests", "assets", "tokenizer")

# The planted secrets. Each is anchored to a distinct prefix so the probe can
# show the model learned *the association*, not just one string it could emit
# regardless of context.
PLANTED = [
    {
        "key": "A",
        "prefix": "the secret launch code is",
        "secret": "ZORBAXQUILL7741",
        "tail": "and the vault door opens only for it",
    },
    {
        "key": "B",
        "prefix": "the backup launch code is",
        "secret": "MIRVANTHOS9932",
        "tail": "and the hatch stays locked without it",
    },
]

N_PLANTED_COPIES = 1500  # per variant
N_FILLER = 40
FILLER_EVERY = 50


def _doc_text(spec: dict) -> str:
    return f"{spec['prefix']} {spec['secret']} {spec['tail']} ."


def _load_filler(n: int) -> list[str]:
    """A few real C4 documents, so the corpus is not 100% repetition."""
    texts: list[str] = []
    with open(C4_TEST, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            texts.append(json.loads(line)["text"])
            if len(texts) >= n:
                break
    return texts


def build(out_dir: str = OUT_DIR, seed: int = 0) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    filler = _load_filler(N_FILLER)

    rows: list[dict] = []
    planted_cycle = [spec["key"] for spec in PLANTED]
    per_variant = N_PLANTED_COPIES
    specs = {spec["key"]: spec for spec in PLANTED}

    counts = {key: 0 for key in specs}
    filler_i = 0
    # Interleave rather than block: the dataloader shards the dataset by index
    # across data-parallel ranks, so a block layout would hand whole ranks
    # nothing but filler.
    while any(counts[key] < per_variant for key in specs):
        for key in planted_cycle:
            if counts[key] >= per_variant:
                continue
            rows.append({"text": _doc_text(specs[key])})
            counts[key] += 1
        if filler_i < len(filler) and len(rows) % FILLER_EVERY == 0:
            rows.append({"text": filler[filler_i]})
            filler_i += 1

    rng.shuffle(rows)

    data_path = os.path.join(out_dir, "data.json")
    with open(data_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = _manifest(rows, data_path)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest


def _manifest(rows: list[dict], data_path: str) -> dict:
    """Record the prompt/expected ids *as the tokenizer produces them*."""
    import cpu_compat

    cpu_compat.apply()

    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    tokenizer = HuggingFaceTokenizer.Config().build(tokenizer_path=TOKENIZER_DIR)

    probes = []
    for spec in PLANTED:
        prompt_ids = tokenizer.encode(spec["prefix"], add_bos=False, add_eos=False)
        secret_ids = tokenizer.encode(spec["secret"], add_bos=False, add_eos=False)
        full_ids = tokenizer.encode(_doc_text(spec), add_bos=True, add_eos=True)
        probes.append(
            {
                "key": spec["key"],
                "prefix_text": spec["prefix"],
                "secret_text": spec["secret"],
                "doc_text": _doc_text(spec),
                "prompt_ids": prompt_ids,
                "secret_ids": secret_ids,
                "prompt_decoded": tokenizer.decode(prompt_ids),
                "secret_decoded": tokenizer.decode(secret_ids),
                "full_doc_ids": full_ids,
            }
        )

    try:
        bos_id = tokenizer.bos_id
        eos_id = tokenizer.eos_id
    except Exception:  # pragma: no cover - the wrapper exposes them as attrs
        bos_id = full_ids[0] if probes else None
        eos_id = None

    return {
        "data_path": os.path.abspath(data_path),
        "n_rows": len(rows),
        "n_planted_per_variant": N_PLANTED_COPIES,
        "n_filler": N_FILLER,
        "tokenizer": os.path.abspath(TOKENIZER_DIR),
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        # Recorded so the probe can reproduce the *exact* context the model was
        # trained on. The corpus feeds each document as [BOS] + text, with
        # positions starting at 0, so a probe that omits BOS asks the question
        # from one position earlier than anything it ever saw.
        "bos_id": bos_id,
        "eos_id": eos_id,
        "probes": probes,
        "corpus_chars": sum(len(row["text"]) for row in rows),
    }


def load_manifest(path: str = MANIFEST_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    info = build()
    print(json.dumps({k: v for k, v in info.items() if k != "probes"}, indent=2))
    for probe in info["probes"]:
        print(
            f"  probe {probe['key']}: prompt={probe['prompt_decoded']!r} "
            f"({len(probe['prompt_ids'])} ids) -> "
            f"secret={probe['secret_decoded']!r} ({len(probe['secret_ids'])} ids)"
        )
