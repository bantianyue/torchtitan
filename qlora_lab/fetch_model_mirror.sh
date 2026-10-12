#!/usr/bin/env bash
# Fetch Qwen3.5-0.8B from hf-mirror.com using N parallel HTTP range requests.
#
# Why: huggingface.co is unreachable/slow from this box (curl gets http 000),
# and the mirror throttles a single connection to ~1.3 MB/s. The weights are one
# 1.747 GB file, so snapshot_download's per-FILE parallelism does not help.
# Range requests ARE supported (206 + Accept-Ranges: bytes), so we split the one
# big file across connections and concatenate.
#
# Output is a flat local model dir, so transformers can load it with
#   --model qlora_lab/model
# and we never depend on the HF cache layout (symlinks are unsupported here).
set -uo pipefail

MIRROR=${MIRROR:-https://hf-mirror.com}
REPO=${REPO:-Qwen/Qwen3.5-0.8B}
LAB=/d/02_source/torchtitan/qlora_lab
DEST="$LAB/model"
BIG=model.safetensors-00001-of-00001.safetensors
SIZE=1746942600
SHA=04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
N=${PARTS:-8}

mkdir -p "$DEST"
echo "mirror=$MIRROR parts=$N dest=$DEST"

echo "--- small files ---"
for f in model.safetensors.index.json config.json tokenizer.json vocab.json \
         merges.txt tokenizer_config.json chat_template.jinja \
         preprocessor_config.json video_preprocessor_config.json; do
  if [ -s "$DEST/$f" ]; then echo "  have $f"; continue; fi
  curl -fsSL --retry 8 --retry-delay 3 -o "$DEST/$f.tmp" "$MIRROR/$REPO/resolve/main/$f" \
    && mv "$DEST/$f.tmp" "$DEST/$f" && echo "  got  $f" \
    || { echo "  FAIL $f"; exit 1; }
done

echo "--- big file ($SIZE bytes, $N ranges) ---"
CHUNK=$(( (SIZE + N - 1) / N ))
pids=()
for i in $(seq 0 $((N - 1))); do
  s=$(( i * CHUNK )); e=$(( s + CHUNK - 1 ))
  [ "$e" -ge "$SIZE" ] && e=$(( SIZE - 1 ))
  [ "$s" -ge "$SIZE" ] && continue
  want=$(( e - s + 1 ))
  part="$DEST/.part.$i"
  if [ -s "$part" ] && [ "$(stat -c %s "$part")" -eq "$want" ]; then
    echo "  part $i cached ($want B)"; continue
  fi
  (
    for try in 1 2 3 4 5 6; do
      rm -f "$part"
      if curl -fsSL -r "$s-$e" --retry 10 --retry-delay 2 -o "$part" \
           "$MIRROR/$REPO/resolve/main/$BIG" \
         && [ "$(stat -c %s "$part" 2>/dev/null || echo 0)" -eq "$want" ]; then
        echo "  part $i ok ($want B, try $try)"; exit 0
      fi
      echo "  part $i retry $try"; sleep 3
    done
    echo "  part $i FAILED"; exit 1
  ) &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "some parts failed"; exit 1; }

echo "--- assemble ---"
rm -f "$DEST/$BIG"
for i in $(seq 0 $((N - 1))); do
  [ -f "$DEST/.part.$i" ] && cat "$DEST/.part.$i" >> "$DEST/$BIG"
done
got=$(stat -c %s "$DEST/$BIG")
echo "assembled: $got bytes (want $SIZE)"
[ "$got" -eq "$SIZE" ] || { echo "SIZE MISMATCH"; exit 1; }

echo "--- verify sha256 (== HF blob etag) ---"
actual=$(sha256sum "$DEST/$BIG" | cut -d' ' -f1)
echo "actual: $actual"
echo "expect: $SHA"
if [ "$actual" != "$SHA" ]; then echo "SHA MISMATCH"; exit 1; fi

rm -f "$DEST"/.part.*
echo "OK -> $DEST"
du -sh "$DEST"
