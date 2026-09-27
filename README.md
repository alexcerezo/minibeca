# SAE TopK sobre `gpt2-small` — capa 8, resid_pre

Sparse Autoencoder **TopK** (`k = 32`, `d_sae = 24 576`, expansión 32×)
entrenado sobre `blocks.8.hook_resid_pre` de `gpt2-small` con **AuxK**
(Gao et al., 2024). Evaluación en held-out, interpretabilidad ciega con
dos evaluadores LLM y control causal por *steering*.

- **Pesos**: [`alexcerezo/sae-gpt2-small-l8-topk32`](https://huggingface.co/alexcerezo/sae-gpt2-small-l8-topk32)
- **Curvas**: [W&B project](https://wandb.ai/alexcerezocontreras-university-of-malaga/minibeca-xmihura) — `rd8xu845` (con AuxK) · `sjdvx9oh` (sin AuxK)
- **Artículo**: <https://alexcerezo.github.io/minibeca/>

## Resultados

Held-out (último shard de OpenWebText, BOS excluido, mismo protocolo en
todos los SAEs; ficheros en [`data/analysis/`](data/analysis)).

| SAE | ΔCE ctx 128 | ΔCE ctx 1024 | CE score 1024 | EV 128 | L₀ |
|---|---:|---:|---:|---:|---:|
| **Este trabajo** (TopK 24k, con AuxK) | **0,119** | **0,143** | **0,983** | 0,850 | 32 |
| Este trabajo, sin AuxK                | 0,280     | 0,325     | 0,961     | 0,760 | 32 |
| OpenAI TopK 32k                       | 0,118     | 0,812     | 0,903     | 0,849 | 32 |
| J. Bloom `res-jb`                     | 0,149     | 0,985     | 0,882     | 0,852 | 67 |

- **Interpretabilidad ciega** (120 latentes vs 120 neuronas MLP L7,
  2 LLMs, ρ = 0,95): latentes **4,34/5** (87 % ≥ 4) frente a neuronas
  **3,09/5** (34 % ≥ 4); Mann-Whitney p ≈ 10⁻²⁰.
- **Steering causal** vs dirección aleatoria de la misma magnitud:
  hasta **55 %** de continuaciones contienen el concepto frente a
  **≤ 3 %** en control.
- Coste: 26 min en un A100 80 GB (~0,70 USD).

## Uso

```python
from huggingface_hub import snapshot_download
from sae_lens import SAE

sae = SAE.load_from_disk(snapshot_download("alexcerezo/sae-gpt2-small-l8-topk32"))
z = sae.encode(x)      # (T, 24_576) activaciones de las latentes
x_hat = sae.decode(z)  # (T, 768) reconstrucción del flujo residual
```

`x` es el flujo residual de `gpt2-small` en `blocks.8.hook_resid_pre`.

## Configuración

| | |
|---|---|
| Modelo | `gpt2-small`, hook `blocks.8.hook_resid_pre` ≡ `blocks.7.hook_resid_post` |
| Arquitectura | TopK, `k = 32`, `d_sae = 24 576`, `rescale_acts_by_decoder_norm=True`, `decoder_init_norm=0.1` |
| AuxK | `k_aux = d_sae / 2 = 12 288`, top-384 sobre latentes muertas, `α = 1/32` |
| Datos | `apollo-research/Skylion007-openwebtext-tokenizer-gpt2` streaming, 100 M tokens, `context_size = 1024`, `train_batch_size_tokens = 8192`, BOS excluido |
| Escalado | constante `α = √d_in / mean‖x‖` sobre 100 batches, plegada con `fold_activation_norm_scaling_factor` al final |
| Optim | Adam(β = 0.9, 0.999, ε = 6.25·10⁻¹⁰), lr `3·10⁻⁴`, warmup 1 000 pasos, cosine último 20 % hasta `lr/10`, grad-clip 1.0 |
| Dead latents | ventana 1 000 pasos (~8 M tokens); rescate exclusivo por AuxK, sin resampling |
| Checkpoints | atómicos con `manifest.json` SHA-256, últimos 3, symlink `latest`, cada 1 000 pasos o 600 s, además al recibir SIGTERM |
| Eval | `sae_lens.run_evals` cada 500 pasos (CE, KL, L₂, sparsity, varianza) + evaluación final antes del upload |
| Infra | Runpod, `runpod-torch-v280` (torch 2.8 + CUDA 12.8 + Python 3.12), A100 80 GB PCIe (o L40S 48 GB), Network Volume 50 GB en `/workspace`, `--terminate-after +3h` |

Justificación completa (comparativa L1/TopK/Gated, auditoría de memoria,
protocolo) en [`docs/PLAN_TECNICO.md`](docs/PLAN_TECNICO.md).

## Reproducir

### Setup

```bash
uv sync
```

### Smoke test (CPU, ~2 min)

Entrena 20 pasos con `gpt2-small` sobre WikiText-103, mata el proceso,
`--resume auto` y confirma continuación bit-exacta:

```bash
make smoke
```

### Análisis held-out

Descarga los pesos desde HF automáticamente. GPU recomendada; en CPU
consume ~6 GB de RAM.

```bash
uv run python src/analyze_sae.py --stage metrics  --n-seqs 512  --ctx 128  --batch 32
uv run python src/analyze_sae.py --stage metrics  --n-seqs 128  --ctx 1024 --batch 8
uv run python src/analyze_sae.py --stage features --n-seqs 4096 --batch 64
uv run python src/analyze_sae.py --stage steer    --latents 10274,19814,19443
```

Salidas en [`data/analysis/`](data/analysis):
`metrics_heldout_ctx{128,1024}.json`, `feature_stats.json`,
`interp_ratings.json`, `steering.json`.

### Entrenamiento completo (Runpod, ~26 min en A100 80 GB)

```bash
bash scripts/provision_pod.sh                        # → POD_ID, POD_IP, POD_PORT, POD_KEY
scp -i "$POD_KEY" -P "$POD_PORT" scripts/*.sh root@$POD_IP:/workspace/
ssh -i "$POD_KEY" -p "$POD_PORT" root@$POD_IP

# Dentro del pod:
export WANDB_API_KEY=... HF_TOKEN=... HF_REPO_ID=<usuario>/sae-gpt2-small-l8-topk32
bash /workspace/setup_pod.sh
bash /workspace/minibeca/scripts/run_tmux.sh

# De vuelta local:
bash scripts/teardown.sh $POD_ID
```

El *watchdog* de `run_tmux.sh` relanza `train_sae.py --resume auto` ante
crashes (OOM transitorio, red, SIGTERM) y termina cuando aparece
`final_inference/eval_metrics.json`.

### Renders

```bash
make videos    # 6 escenas Manim → assets/videos/
make covers    # 5 portadas + prisma Manim → assets/cover/
```

## Estructura

```
minibeca/
├── src/
│   ├── train_sae.py           # loop TopK + AuxK, checkpoints atómicos, --resume auto
│   ├── analyze_sae.py         # métricas held-out, features, steering causal
│   └── viz/                   # 6 escenas Manim + portadas
├── scripts/                   # provision_pod, setup_pod, run_tmux (watchdog), teardown
├── data/
│   ├── analysis/              # salidas de analyze_sae.py
│   ├── wandb/                 # curvas exportadas con y sin AuxK
│   └── manim_data.json        # resumen consumido por las animaciones
├── assets/
│   ├── videos/                # 6 escenas 1080p60
│   └── cover/                 # portadas
├── docs/
│   ├── index.md               # artículo publicado en Pages
│   ├── _config.yml            # config de Jekyll
│   ├── assets/videos/         # copia servida por Pages
│   ├── PLAN_TECNICO.md        # arquitectura, memoria, protocolo, infra
│   ├── articulo/              # versiones del artículo (v1, v2, v3)
│   └── papers/                # papers de referencia (HTML)
├── notebooks/                 # prueba de concepto en Colab
├── Makefile                   # smoke · videos · covers · clean
├── pyproject.toml             # deps (uv)
└── uv.lock
```

`outputs/` (pesos locales de la ablación sin AuxK, 145 MB) y `build/`
(cachés de Manim) están ignorados; se regeneran o se consumen desde HF /
W&B.

## Referencias

1. Elhage et al., [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html), Anthropic, 2022.
2. Gao et al., [*Scaling and evaluating sparse autoencoders*](https://cdn.openai.com/papers/sparse-autoencoders.pdf), OpenAI, 2024. TopK + AuxK.
3. Bricken et al., [*Towards Monosemanticity*](https://transformer-circuits.pub/2023/monosemantic-features/index.html), Anthropic, 2023. Escala 1–5 de interpretabilidad.
4. Templeton et al., [*Scaling Monosemanticity*](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html), Anthropic, 2024.
5. Bloom, [*SAELens*](https://github.com/jbloomAus/SAELens).
