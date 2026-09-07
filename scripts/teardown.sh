#!/usr/bin/env bash
# teardown.sh — Verifica que el artefacto final está en Hugging Face (o en el
# volumen) y libera la GPU. El volumen persiste con el modelo hasta que se
# borre explícitamente.

set -euo pipefail

POD_ID="${1:-${POD_ID:-}}"
: "${POD_ID:?POD_ID requerido: bash scripts/teardown.sh <pod-id>}"

command -v runpodctl >/dev/null || { echo "runpodctl no instalado" >&2; exit 1; }

if [[ -n "${HF_REPO_ID:-}" ]]; then
    echo "==> comprobando presencia de sae_weights.safetensors en HF: $HF_REPO_ID"
    python3 - <<PY
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files("$HF_REPO_ID")
assert "sae_weights.safetensors" in files, f"missing weights: {files}"
print("OK — artefacto presente en HF")
PY
fi

echo "==> borrando pod ${POD_ID} (el volumen persiste)"
runpodctl pod remove "${POD_ID}"

echo "==> volúmenes vivos:"
runpodctl network-volume list
