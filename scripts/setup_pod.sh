#!/usr/bin/env bash
# setup_pod.sh — Se ejecuta DENTRO del pod (por SSH). Deja el entorno listo.
#
# Uso remoto:
#   scp -i "$POD_KEY" -P "$POD_PORT" scripts/setup_pod.sh root@$POD_IP:/workspace/
#   ssh -i "$POD_KEY" -p "$POD_PORT" root@$POD_IP 'bash /workspace/setup_pod.sh'

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/aleja/minibeca.git}"
REPO_DIR="${REPO_DIR:-/workspace/minibeca}"

# 1) Caches y trabajos permanentes en el network volume (persisten al borrar el pod)
export HF_HOME=/workspace/hf-cache
export WANDB_DIR=/workspace/wandb
mkdir -p "$HF_HOME" "$WANDB_DIR" /workspace/checkpoints /workspace/logs
grep -q 'HF_HOME=/workspace/hf-cache' /root/.bashrc || cat >> /root/.bashrc <<'EOF'
export HF_HOME=/workspace/hf-cache
export WANDB_DIR=/workspace/wandb
EOF

# 2) PEP 668: torch vive en el Python del sistema (3.12); NO crear venv nuevo.
python3 -V
pip install --break-system-packages -q --upgrade \
    "sae-lens>=6.51" "transformer-lens>=2.15" wandb huggingface_hub datasets safetensors

# 3) Autenticación no interactiva (si el usuario exportó los tokens antes del ssh)
if [[ -n "${HF_TOKEN:-}" ]]; then
    python3 -c "from huggingface_hub import HfFolder; HfFolder.save_token('$HF_TOKEN')"
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login --relogin --host=https://api.wandb.ai "$WANDB_API_KEY" >/dev/null
fi

# 4) Repo
if [[ ! -d "$REPO_DIR/.git" ]]; then
    git clone "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only || true
fi

# 5) Sanity check GPU + versiones
python3 - <<'PY'
import torch, sae_lens, transformer_lens
print("torch     :", torch.__version__, "cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("sae_lens  :", sae_lens.__version__)
print("tl        :", transformer_lens.__version__)
PY

echo "==> pod listo. Lanza el entrenamiento con: bash scripts/run_tmux.sh"
