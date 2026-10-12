"""Introspect Qwen3.5-0.8B without allocating weights.

Prints:
  - text_config / vision_config key geometry
  - parameter counts (total, language model, vision tower)
  - every nn.Linear name, grouped, so LoRA target_modules can be chosen exactly
"""
import os, sys, collections

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf"))

import torch
import torch.nn as nn
from transformers import AutoConfig

_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL = os.environ.get("QWEN_MODEL") or (_LOCAL if os.path.isdir(_LOCAL)
                                         else "Qwen/Qwen3.5-0.8B")

cfg = AutoConfig.from_pretrained(MODEL)
tc = cfg.text_config
vc = cfg.vision_config

print("=== text_config ===")
for k in [
    "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    "head_dim", "vocab_size", "intermediate_size", "tie_word_embeddings",
    "max_position_embeddings", "layer_types", "rope_parameters",
]:
    v = getattr(tc, k, None)
    print(f"  {k:24s} = {str(v)[:220]}")

print("=== vision_config ===")
for k in ["depth", "hidden_size", "out_hidden_size", "num_heads", "patch_size", "in_channels"]:
    print(f"  {k:24s} = {str(getattr(vc, k, None))[:120]}")

print("=== build model on meta (no weights) ===")
from accelerate import init_empty_weights
from transformers import AutoModelForImageTextToText

with init_empty_weights():
    model = AutoModelForImageTextToText.from_config(cfg)

def count(mod):
    return sum(p.numel() for p in mod.parameters())

print(f"  total      : {count(model):,}")
names = dict(model.named_modules())
lm_prefix = None
for cand in ("model.language_model", "language_model", "model.model"):
    if cand in names:
        lm_prefix = cand
        break
print("  lm_prefix  :", lm_prefix)
if lm_prefix:
    print(f"  language   : {count(names[lm_prefix]):,}")
for cand in ("model.visual", "visual", "model.vision_tower"):
    if cand in names:
        print(f"  vision     : {count(names[cand]):,}  ({cand})")

print("=== nn.Linear inventory (by leaf name) ===")
buckets = collections.Counter()
samples = {}
for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear):
        leaf = name.split(".")[-1]
        buckets[leaf] += 1
        if leaf not in samples:
            samples[leaf] = (name, mod.in_features, mod.out_features)
for leaf, n in buckets.most_common():
    full, i, o = samples[leaf]
    print(f"  {leaf:22s} x{n:<5d} e.g. {full}  [{i} -> {o}]")

print("=== first decoder layer module tree ===")
if lm_prefix:
    layers = names.get(lm_prefix + ".layers")
    if layers is not None:
        n0 = layers[0]
        for name, mod in n0.named_modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                print(f"    {name:44s} {type(mod).__name__:10s} {tuple(mod.weight.shape)}")
print("layer_types (first 9):", str(getattr(tc, "layer_types", None))[:300])
