#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/backend"

models=(
  "DecAttn"
  "Decoder"
  "EncAttn"
  "encdec_crossattn_swiglu"
  "encoder_alibi_prenorm"
  "falcon7b"
  "FFN"
  "keyformer_prenorm"
  "keyformer"
  "llama3_8b_gqa"
  "llama8b"
  "prenorm"
  "qknorm_prenorm"
  "qknorm"
  "rmsnorm"
  "Roco_prenorm"
  "Roco"
)

for model_name in "${models[@]}"; do
  echo "==> ${model_name}"
  python -m profile.benchmark \
    --ir "../frontend/outputs/trinity/${model_name}/ir.txt" \
    --shapes "../frontend/outputs/trinity/${model_name}/shapes.json"
  echo
done
