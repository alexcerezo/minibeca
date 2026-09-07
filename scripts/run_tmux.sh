#!/usr/bin/env bash
# run_tmux.sh — Lanza en el pod:
#   ventana 0: training loop supervisado (reinicia con --resume auto si cae)
#   ventana 1: telemetría GPU (nvidia-smi -> CSV en /workspace/logs/gpu.csv)
#   ventana 2: tail del log de entrenamiento
#
# Sobrevive a desconexiones SSH (todo dentro de tmux). Si el pod se reinicia,
# basta con volver a lanzar este script; `--resume auto` reanuda bit-exacto.

set -euo pipefail

SESSION="${SESSION:-sae}"
REPO_DIR="${REPO_DIR:-/workspace/minibeca}"
CKPT_DIR="${CKPT_DIR:-/workspace/checkpoints/sae-gpt2}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"

cd "$REPO_DIR"

# --- ventana de entrenamiento (loop supervisado) ---
TRAIN_CMD=$(cat <<EOF
set -e
export HF_HOME=/workspace/hf-cache
cd $REPO_DIR
while true; do
    ts=\$(date -Is)
    echo "[watchdog \$ts] launching train_sae.py"
    python3 -u src/train_sae.py \\
        --checkpoint-dir $CKPT_DIR \\
        --resume auto \\
        --precision bf16 \\
        --wandb-project minibeca-xmihura \\
        --wandb-mode \${WANDB_MODE:-online} \\
        \${HF_REPO_ID:+--hf-repo-id \$HF_REPO_ID} \\
        \${EXTRA_ARGS:-} 2>&1 | tee -a $LOG_DIR/train.log
    rc=\${PIPESTATUS[0]}
    echo "[watchdog] training exited rc=\$rc"
    # Si terminó porque llegó a total_tokens, salir; si crasheó, reintentar.
    if [[ -f $CKPT_DIR/final_inference/eval_metrics.json ]]; then
        echo "[watchdog] final_inference presente ⇒ done, saliendo."
        exit 0
    fi
    echo "[watchdog] backoff 30s antes de reintentar…"
    sleep 30
done
EOF
)

# --- telemetría GPU ---
TELEMETRY_CMD=$(cat <<EOF
CSV=$LOG_DIR/gpu.csv
if [[ ! -f \$CSV ]]; then
    echo "timestamp,idx,name,util_gpu,mem_used_mb,mem_total_mb,temp_c,power_w" > \$CSV
fi
while true; do
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \\
               --format=csv,noheader,nounits >> \$CSV
    sleep 15
done
EOF
)

tmux has-session -t "$SESSION" 2>/dev/null && {
    echo "sesión $SESSION ya existe; usa: tmux attach -t $SESSION"; exit 0; }

tmux new-session -d -s "$SESSION" -n train "bash -lc '$TRAIN_CMD'"
tmux new-window -t "$SESSION" -n gpu     "bash -lc '$TELEMETRY_CMD'"
tmux new-window -t "$SESSION" -n log     "tail -F $LOG_DIR/train.log"

echo "tmux session '$SESSION' arrancada. Adjúntate con:"
echo "  tmux attach -t $SESSION"
