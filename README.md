# Mini-Beca XMihura — Sparse Autoencoder sobre `gpt2-small`

Entregable de la convocatoria **Minibecas XMihura**. Entrena un **Sparse
Autoencoder TopK** (`k = 32`, `d_sae = 24 576`, expansión 32×) sobre
`blocks.8.hook_resid_pre` de `gpt2-small`, con AuxK, checkpointing atómico y
reanudación bit-exacta; después evalúa reconstrucción held-out,
interpretabilidad ciega (LLM-as-judge) y control causal por *steering*.

- **Pesos** → [`alexcerezo/sae-gpt2-small-l8-topk32`](https://huggingface.co/alexcerezo/sae-gpt2-small-l8-topk32) (HF Hub)
- **Curvas** → [W&B project `minibeca-xmihura`](https://wandb.ai/alexcerezocontreras-university-of-malaga/minibeca-xmihura) — `rd8xu845` (con AuxK) · `sjdvx9oh` (sin AuxK)
- **Artículo** → [`docs/articulo/ARTICULO_X_v3.md`](docs/articulo/ARTICULO_X_v3.md)
- **Vídeos y portadas** → [`assets/videos/`](assets/videos), [`assets/cover/`](assets/cover)

## Resultados

Evaluación en datos **held-out** (último shard de OpenWebText, BOS
excluido, mismo protocolo para todos los SAEs; ficheros en
[`data/analysis/`](data/analysis)):

| SAE | ΔCE ctx 128 | ΔCE ctx 1024 | CE score 1024 | EV 128 | L₀ |
|---|---:|---:|---:|---:|---:|
| **Este trabajo** (TopK 24k, con AuxK) | **0,119** | **0,143** | **0,983** | 0,850 | 32 |
| Este trabajo, sin AuxK                | 0,280     | 0,325     | 0,961     | 0,760 | 32 |
| OpenAI TopK 32k                       | 0,118     | 0,812     | 0,903     | 0,849 | 32 |
| J. Bloom `res-jb`                     | 0,149     | 0,985     | 0,882     | 0,852 | 67 |

- **Interpretabilidad ciega** (2 evaluadores LLM independientes, ρ = 0,95;
  120 latentes vs 120 neuronas MLP L7): latentes **4,34/5** (87 % ≥ 4)
  frente a neuronas **3,09/5** (34 % ≥ 4).
- **Steering causal** con control aleatorio: hasta **55 %** de textos
  generados contienen el concepto objetivo frente a **≤ 3 %** en control.
- **Coste**: 26 min en un A100 80 GB, ~0,69 USD.

Detalle numérico en `data/analysis/metrics_heldout_ctx{128,1024}.json`,
`feature_stats.json`, `interp_ratings.json`, `steering.json`.

## Estructura

```
minibeca/
├── src/
│   ├── train_sae.py            # loop TopK + AuxK, checkpoints atómicos, --resume auto
│   ├── analyze_sae.py          # métricas held-out, features, steering causal
│   └── viz/
│       ├── manim_scenes.py     # 6 escenas del artículo (Manim)
│       ├── cover.py            # 5 portadas (cairo + UMAP del decoder)
│       └── manim_cover.py      # portada "prisma" (Manim)
├── scripts/                    # ciclo GPU (Runpod)
│   ├── provision_pod.sh        # crea volumen + pod, imprime credenciales
│   ├── setup_pod.sh            # dependencias + auth (HF, W&B) dentro del pod
│   ├── run_tmux.sh             # tmux con watchdog + telemetría GPU
│   └── teardown.sh             # verifica artefacto en HF y libera el pod
├── data/
│   ├── analysis/               # salidas de analyze_sae.py
│   ├── wandb/                  # curvas exportadas (train/eval, con y sin AuxK)
│   ├── cover/                  # UMAP del decoder + activaciones para portadas
│   └── manim_data.json         # resumen que consumen las animaciones
├── assets/                     # productos finales listos para publicar
│   ├── videos/                 # 6 escenas 1080p60
│   ├── cover/                  # 5 portadas + prisma Manim
│   └── archive/                # vídeo y miniaturas de una versión anterior
├── docs/
│   ├── PLAN_TECNICO.md         # arquitectura, memoria, protocolo, infra
│   ├── ESTADO.md               # estado del proyecto / handoff
│   ├── articulo/               # v3 (final), v2 (divulgativo), v1 (técnico)
│   └── papers/                 # papers de referencia (HTML)
├── notebooks/
│   └── sae_replication_colab.ipynb   # prueba de concepto previa
├── Makefile                    # make smoke · videos · covers · clean
├── pyproject.toml              # dependencias (uv)
└── uv.lock
```

`outputs/` (pesos locales de la ablación sin AuxK, 145 MB) y `build/`
(cachés de render de Manim) están ignorados; se regeneran a demanda o se
consumen directamente desde HF Hub / W&B.

## Reproducibilidad

### Setup local

```bash
uv sync
```

### Smoke test (CPU, ~2 min)

Entrena 20 pasos con `gpt2-small` real sobre WikiText-103, escribe un
checkpoint, lo mata, hace `--resume auto` y confirma continuación
bit-exacta:

```bash
make smoke
```

### Análisis held-out

GPU recomendada; en CPU consume ~6 GB de RAM. Descarga los pesos
publicados desde HF Hub automáticamente:

```bash
uv run python src/analyze_sae.py --stage metrics  --n-seqs 512  --ctx 128  --batch 32
uv run python src/analyze_sae.py --stage metrics  --n-seqs 128  --ctx 1024 --batch 8
uv run python src/analyze_sae.py --stage features --n-seqs 4096 --batch 64
uv run python src/analyze_sae.py --stage steer    --latents 10274,19814,19443
```

### Renders

```bash
make videos    # 6 escenas Manim → assets/videos/
make covers    # 5 portadas + prisma Manim → assets/cover/
```

### Entrenamiento completo en GPU alquilada (Runpod)

```bash
# 1) Provisionar (imprime POD_ID, POD_IP, POD_PORT, POD_KEY)
bash scripts/provision_pod.sh

# 2) Copiar scripts y entrar al pod
scp -i "$POD_KEY" -P "$POD_PORT" scripts/*.sh root@$POD_IP:/workspace/
ssh -i "$POD_KEY" -p "$POD_PORT" root@$POD_IP

# 3) Dentro del pod: credenciales + arranque
export WANDB_API_KEY=...
export HF_TOKEN=...
export HF_REPO_ID=<usuario>/sae-gpt2-small-l8-topk32
bash /workspace/setup_pod.sh
bash /workspace/minibeca/scripts/run_tmux.sh
tmux attach -t sae             # opcional; Ctrl-b d para salir

# 4) Verificar el artefacto en HF y liberar la GPU
bash scripts/teardown.sh $POD_ID
```

El *watchdog* de `run_tmux.sh` relanza `train_sae.py --resume auto` ante
cualquier crash (OOM transitorio, pérdida de red, SIGTERM del scheduler)
y termina limpiamente cuando aparece `final_inference/eval_metrics.json`.

## Configuración fija

| | |
|---|---|
| Modelo base | `gpt2-small`, hook `blocks.8.hook_resid_pre` (≡ `blocks.7.hook_resid_post`) |
| Arquitectura SAE | **TopK**, `k = 32`, `d_sae = 24 576` (expansión 32×), `rescale_acts_by_decoder_norm=True`, `decoder_init_norm=0.1` |
| Pérdida auxiliar | **AuxK** (Gao et al. 2024) con `α = 1/32`, `k_aux = d_sae / 2` |
| Datos | `apollo-research/Skylion007-openwebtext-tokenizer-gpt2` streaming, 100 M tokens, `context_size=1024`, `train_batch_size_tokens=8192`, BOS excluido |
| Escalado | constante `α = √d_in / mean‖x‖` estimada sobre 100 batches, plegada en los pesos al final |
| Optim | Adam(β=(0.9, 0.999), ε=6.25·10⁻¹⁰), lr `3·10⁻⁴`, warmup 1 000 pasos, cosine en el último 20 % hasta `lr/10`, grad-clip 1.0 |
| Checkpoints | atómicos con `manifest.json` SHA-256, últimos 3 conservados, symlink `latest`, cada 1 000 pasos o 600 s + al recibir SIGTERM |
| Eval | `sae_lens.run_evals` cada 500 pasos (CE, KL, L₂, sparsity, varianza) + evaluación final antes del upload |
| Infra | Runpod, template `runpod-torch-v280` (torch 2.8 + CUDA 12.8 + Python 3.12), A100 80 GB PCIe (o L40S 48 GB), Network Volume 50 GB en `/workspace`, cost-guard `--terminate-after +3h` |

Justificación completa (comparativa L1/TopK/Gated, auditoría de memoria,
protocolo) en [`docs/PLAN_TECNICO.md`](docs/PLAN_TECNICO.md).

## Conceptos clave

- **Polisemanticidad** — una misma neurona responde a conceptos distintos.
- **Superposición** — el modelo representa más *features* que dimensiones,
  proyectándolas sobre direcciones casi-ortogonales.
- **Sparse Autoencoder (TopK)** — diccionario sobrecompleto que
  descompone las activaciones residuales en *features* aproximadamente
  monosemánticas; `k = 32` fija `L₀` por construcción y elimina el
  hiperparámetro `λ`.
- **AuxK loss** — las latentes muertas se rescatan aprendiendo a
  reconstruir el residuo de las vivas; en esta corrida rescata **21 738
  de 21 780** latentes muertas entre los pasos 1 200 y 5 000.

## Referencias

En `docs/papers/` (HTML íntegros):

1. Elhage et al., *Toy Models of Superposition* (Anthropic, 2022).
2. Bricken et al., *Towards Monosemanticity: Decomposing Language Models
   With Dictionary Learning* (Anthropic, 2023).
3. Templeton et al., *Scaling Monosemanticity: Extracting Interpretable
   Features from Claude 3 Sonnet* (Anthropic, 2024).
4. Gao et al., *Scaling and evaluating sparse autoencoders* (OpenAI, 2024)
   — origen de TopK + AuxK.

## Licencia

Código bajo MIT. Pesos publicados en HF Hub bajo la licencia del
repositorio correspondiente.
