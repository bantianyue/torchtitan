# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prove the trained checkpoint learned the planted secrets.

A falling loss is weak evidence -- it is consistent with learning byte
frequencies and nothing else. This probe asks a question only a model that
actually read the corpus can answer: *what is the launch code?*

For each planted secret it runs the same three measurements on two models:

- **trained**: built from the run's ``--config`` (so the architecture is
  identical by construction) and loaded from the DCP checkpoint the run wrote.
- **random init**: the same architecture, freshly initialised. Everything
  "learned" has to show up as a difference against this.

Measurements, per secret:

1. ``p_secret`` -- the probability the model assigns to the *entire* secret
   string, given the prompt, under teacher forcing. This is the headline
   number: exp(-sum of per-token NLL). A model that has memorised the string
   scores near 1; a model that has not scores near 1/vocab per token, which for
   a 20-character secret over a 2020-token vocabulary is astronomically small.
2. ``accuracy`` -- how many of the secret's tokens are the argmax when the
   model is fed the prompt and the secret's own prefix.
3. ``greedy`` -- the string the model actually generates, so the result can be
   read without interpreting a probability.

Also reported: the teacher-forced mean NLL on the whole planted document, which
shows *where* the learning happened (the prompt region is only a few tokens, the
secret is the part that cannot be guessed).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import cpu_compat

cpu_compat.apply()

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_MANIFEST = os.path.join(HERE, "probe_manifest.json")


def _load_checkpoint(model, ckpt_dir: str) -> str:
    """Load a DCP checkpoint written by a run of any parallel strategy.

    The saved files are per-rank shards, but a checkpoint is a *whole* model
    regardless: DCP merges the shards, so a single process can reconstruct the
    full state dict even if the run used 8 ranks. That is what makes it possible
    to probe a tensor-parallel or pipeline-parallel run from one process.
    """
    import glob

    import torch.distributed.checkpoint as dcp

    shards = sorted(glob.glob(os.path.join(ckpt_dir, "*.distcp")))
    if not shards:
        raise FileNotFoundError(f"no .distcp shards under {ckpt_dir}")
    state = model.state_dict()
    dcp.load(state, checkpoint_id=ckpt_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match this architecture: "
            f"missing={missing} unexpected={unexpected}"
        )
    return f"{len(shards)} shard(s)"


@torch.no_grad()
def _score_secret(model, prompt_ids: list[int], secret_ids: list[int]) -> dict:
    """Teacher-forced probability, accuracy and greedy output for one secret."""
    import math

    prompt = torch.tensor([prompt_ids], dtype=torch.long)
    n_prompt, n_secret = len(prompt_ids), len(secret_ids)

    # Positions reset per document in the training data, so the probe continues
    # the same numbering: prompt occupies 0..n_prompt-1, the secret follows.
    full_ids = prompt_ids + secret_ids
    positions = torch.arange(len(full_ids), dtype=torch.long).unsqueeze(0)
    logits = model(torch.tensor([full_ids], dtype=torch.long), positions).float()

    # Logits at index i predict token i+1.
    predict_from = n_prompt - 1
    window = logits[0, predict_from : predict_from + n_secret]
    targets = torch.tensor(secret_ids, dtype=torch.long)

    # Everything probabilistic is done in float64: over 15 tokens, float32
    # rounding in log_softmax is enough to make a product of probabilities
    # exceed 1, which reads as a bug in the model rather than in the probe.
    probs64 = torch.log_softmax(window.double(), dim=-1)
    logp_target = probs64.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    per_token_nll = -logp_target
    total_nll = float(per_token_nll.sum())
    # exp(-total_nll) underflows to 0 below ~1e-308, which is the right answer
    # for a model that has no idea what the secret is; keep it exact instead.
    p_secret = math.exp(-total_nll) if total_nll < 700 else 0.0
    argmax_ids = window.argmax(dim=-1).tolist()

    # Greedy decode: feed the prompt, then append the model's own argmax.
    generated = list(prompt_ids)
    greedy_logp = []
    for _ in range(n_secret):
        pos = torch.arange(len(generated), dtype=torch.long).unsqueeze(0)
        out = model(torch.tensor([generated], dtype=torch.long), pos).float()
        step_logp = torch.log_softmax(out[0, -1].double(), dim=-1)
        nxt = int(out[0, -1].argmax())
        greedy_logp.append(float(step_logp[nxt]))
        generated.append(nxt)
    greedy_nll = -sum(greedy_logp)

    first_probs = torch.softmax(window[0].double(), dim=-1)
    rank_of_first = int((first_probs > first_probs[secret_ids[0]]).sum()) + 1

    return {
        "prompt_len": n_prompt,
        "secret_len": n_secret,
        "p_secret": p_secret,
        "p_secret_log10": float(math.log10(p_secret)) if p_secret > 0 else -math.inf,
        "total_nll": total_nll,
        "mean_nll_on_secret": float(per_token_nll.mean()),
        "accuracy": float((torch.tensor(argmax_ids) == targets).float().mean()),
        "argmax_ids": argmax_ids,
        "p_first_token": float(first_probs[secret_ids[0]]),
        "rank_first_token": rank_of_first,
        "greedy_ids": generated[n_prompt:],
        "p_secret_full_greedy": math.exp(-greedy_nll) if greedy_nll < 700 else 0.0,
    }


@torch.no_grad()
def _score_document(model, doc_ids: list[int]) -> dict:
    """Mean NLL over a whole planted document, prompt and secret alike."""
    x = torch.tensor([doc_ids], dtype=torch.long)
    positions = torch.arange(len(doc_ids), dtype=torch.long).unsqueeze(0)
    logits = model(x, positions).float()[0, :-1]
    logprobs = torch.log_softmax(logits.double(), dim=-1)
    targets = x[0, 1:]
    nll = -logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    argmax = logits.argmax(dim=-1)
    return {
        "doc_len": len(doc_ids),
        "mean_nll": float(nll.mean()),
        "accuracy": float((argmax == targets).float().mean()),
    }


def _decoded(tokenizer, ids: list[int]) -> str:
    try:
        return tokenizer.decode(list(ids))
    except Exception:  # pragma: no cover - decode is best effort for display
        return "<undecodable>"


def run(
    config_fn: str,
    module: str,
    checkpoint: str,
    out_path: str,
    manifest_path: str = DEFAULT_MANIFEST,
    seed: int = 0,
) -> dict:
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    config_registry = importlib.import_module(f"{module}.config_registry")
    trainer_config = getattr(config_registry, config_fn)()
    spec = trainer_config.model_spec

    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    tokenizer = HuggingFaceTokenizer.Config().build(
        tokenizer_path=trainer_config.hf_assets_path
    )

    models: dict[str, object] = {}
    for label in ("random_init", "trained"):
        torch.manual_seed(seed)
        model = spec.model.build()
        model.init_states()
        model.eval()
        models[label] = model

    checkpoint_note = _load_checkpoint(models["trained"], checkpoint)
    # Weight tying: ``config.enable_weight_tying`` decides whether the token
    # embedding and the output head *are* the same tensor. When they are,
    # ``load_state_dict`` has already written the same values into both names.
    # When they are not (every PP config uses an untied flavor, because a tied
    # parameter cannot live on two pipeline stages), re-pointing one at the other
    # would **replace the trained output head with the embedding matrix** and the
    # model would emit noise while looking perfectly healthy -- so the tying flag
    # is checked rather than assumed.
    tied = bool(getattr(spec.model, "enable_weight_tying", False))
    trained = models["trained"]
    if tied and getattr(trained, "lm_head", None) is not None:
        tok = getattr(getattr(trained, "tok_embeddings", None), "tok", None)
        if tok is not None:
            trained.lm_head.weight = tok.weight

    results: dict[str, dict] = {"probes": [], "documents": []}
    bos_id = manifest.get("bos_id")
    for probe in manifest["probes"]:
        # The corpus feeds every document as [BOS] + text with positions from 0,
        # so the prompt is presented the same way. Skipping BOS would ask the
        # model the question one position earlier than anything it ever saw --
        # which costs accuracy for a model with learned absolute positions and
        # looks like a weak checkpoint rather than a flawed probe.
        prompt_ids = ([bos_id] if bos_id is not None else []) + probe["prompt_ids"]
        entry = {
            "key": probe["key"],
            # Key names follow planted_corpus.py's manifest: the prefix is the
            # text *before* the secret, which is what the model is given.
            "prompt_text": probe["prefix_text"],
            "prompt_decoded": probe["prompt_decoded"],
            "secret_text": probe["secret_text"],
            "secret_decoded": probe["secret_decoded"],
            "prompt_ids": prompt_ids,
        }
        for label, model in models.items():
            scored = _score_secret(model, prompt_ids, probe["secret_ids"])
            scored["argmax_text"] = _decoded(tokenizer, scored["argmax_ids"])
            scored["greedy_text"] = _decoded(tokenizer, scored["greedy_ids"])
            scored["exact_match"] = scored["greedy_ids"] == probe["secret_ids"]
            entry[label] = scored
        results["probes"].append(entry)

    for probe in manifest["probes"]:
        doc_entry = {"key": probe["key"]}
        for label, model in models.items():
            doc_entry[label] = _score_document(model, probe["full_doc_ids"])
        results["documents"].append(doc_entry)

    payload = {
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_loaded": checkpoint_note,
        "config": f"{module}.{config_fn}",
        "flavor": spec.flavor,
        "manifest": os.path.abspath(manifest_path),
        "corpus_rows": manifest["n_rows"],
        "vocab_size": manifest["tokenizer_vocab_size"],
        "results": results,
    }

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="directory with .distcp shards")
    parser.add_argument("--out", default="", help="where to write the probe JSON")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    payload = run(
        config_fn=args.config,
        module=args.module,
        checkpoint=args.checkpoint,
        out_path=args.out,
        manifest_path=args.manifest,
    )

    print("=" * 78)
    print("INFERENCE PROBE -- does the trained model know the planted secrets?")
    print("=" * 78)
    for entry in payload["results"]["probes"]:
        before, after = entry["random_init"], entry["trained"]
        print(f"\n  secret {entry['key']}: {entry['secret_text']!r}")
        print(f"    prompt            {entry['prompt_decoded']!r}")
        print(
            f"    p(secret)         random={before['p_secret']:.3e}"
            f"   trained={after['p_secret']:.3e}"
        )
        print(
            f"    token accuracy    random={before['accuracy']:.2f}"
            f"        trained={after['accuracy']:.2f}"
        )
        print(f"    greedy (random)   {before['greedy_text']!r}")
        print(f"    greedy (trained)  {after['greedy_text']!r}")
        print(f"    exact match       {after['exact_match']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
