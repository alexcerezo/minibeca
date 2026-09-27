# Mini-Beca XMihura — Sparse Autoencoder sobre `gpt2-small`

Entregable para la convocatoria de las **Minibecas XMihura**. Entrena un
Sparse Autoencoder TopK sobre la capa `blocks.8.hook_resid_pre` de
`gpt2-small`, con checkpointing atómico y reanudación bit-exacta.

> **Resultados en datos held-out** (último shard de OpenWebText, BOS excluido,
> mismo protocolo para todos los SAEs; `data/analysis/`):
>
> | SAE | ΔCE ctx 128 | ΔCE ctx 1024 | CE score 1024 | EV 128 | L₀ |
> |---|---:|---:|---:|---:|---:|
> | **Este trabajo** (TopK 24k) | **0,119** | **0,143** | **0,983** | 0,850 | 32 |
> | Este trabajo, sin AuxK | 0,280 | 0,325 | 0,961 | 0,760 | 32 |
> | OpenAI TopK 32k | 0,118 | 0,812 | 0,903 | 0,849 | 32 |
> | J. Bloom res-jb | 0,149 | 0,985 | 0,882 | 0,852 | 67 |
>
> Evaluación ciega (2 evaluadores LLM, ρ = 0,95): latentes 4,34/5 (87 % ≥ 4)
> frente a neuronas MLP L7 3,09/5 (34 % ≥ 4). Steering con control aleatorio:
> hasta 55 % de textos con el concepto frente a ≤ 3 %.
>
> Curvas → [W&B](https://wandb.ai/alexcerezocontreras-university-of-malaga/minibeca-xmihura)
> (`rd8xu845` con AuxK, `sjdvx9oh` sin AuxK) · Pesos → [HF Hub](https://huggingface.co/alexcerezo/sae-gpt2-small-l8-topk32)
> · Artículo → [`docs/articulo/ARTICULO_X_v3.md`](docs/articulo/ARTICULO_X_v3.md) (versiones anteriores en `docs/articulo/`).

## 📚 Referencias

En `docs/papers/` (papers HTML de Anthropic):

1. *Toy Models of Superposition* (2022)
2. *Towards Monosemanticity* (2023)
3. *Scaling Monosemanticity* (2024)

Comparativa y decisiones de diseño → **`docs/PLAN_TECNICO.md`** (bloques
1-3 y 5). Estado del proyecto → **`docs/ESTADO.md`**.

## 🗂 Estructura

```
minibeca/
├── assets/                # productos finales, listos para publicar
│   ├── videos/            # 6 escenas del artículo (1080p60)
│   ├── cover/             # portadas (5 en cairo + prisma en Manim)
│   └── archive/           # vídeo y miniaturas de una versión anterior
├── docs/
│   ├── articulo/          # ARTICULO_X_v3.md (actual), v2 (divulgativo con analogías), v1 (técnico)
│   ├── PLAN_TECNICO.md    # arquitectura, memoria, protocolo, infra
│   ├── ESTADO.md          # estado del proyecto / handoff
│   └── papers/            # papers de referencia (HTML)
├── data/
│   ├── analysis/          # salidas de analyze_sae.py (métricas, features, steering, ratings)
│   ├── wandb/             # curvas de W&B con y sin AuxK (train_*.csv, eval_*.csv)
│   ├── cover/             # datos derivados para las portadas (UMAP del decoder, activaciones)
│   └── manim_data.json    # resumen que consumen las animaciones
├── src/
│   ├── train_sae.py       # loop de entrenamiento (checkpoints atómicos, resume, --aux-coef)
│   ├── analyze_sae.py     # evaluación held-out, features, steering causal
│   └── viz/
│       ├── manim_scenes.py   # 6 escenas del artículo
│       ├── cover.py          # portadas (cairo)
│       └── manim_cover.py    # portada prisma (Manim)
├── scripts/
│   ├── provision_pod.sh   # crea volumen + pod GPU
│   ├── setup_pod.sh       # dependencias + auth dentro del pod
│   ├── run_tmux.sh        # tmux con watchdog + telemetría GPU
│   └── teardown.sh        # verifica artefacto en HF y libera GPU
├── notebooks/
│   └── sae_replication_colab.ipynb   # prueba de concepto previa (Colab)
├── outputs/               # pesos locales de la ablación sin AuxK (ignorado)
├── build/                 # cachés de render de Manim (ignorado, desechable)
├── Makefile               # make videos · make covers · make smoke · make clean
├── pyproject.toml         # uv
└── README.md
```

## 🚀 Ciclo de trabajo

### 1. Provisionar y ejecutar en la GPU alquilada

```bash
# En tu máquina
bash scripts/provision_pod.sh          # imprime POD_ID, POD_IP, POD_PORT, POD_KEY

# Copia el setup y arranca sesión SSH
scp -i "$POD_KEY" -P "$POD_PORT" scripts/*.sh root@$POD_IP:/workspace/
ssh -i "$POD_KEY" -p "$POD_PORT" root@$POD_IP

# Dentro del pod
export WANDB_API_KEY=...
export HF_TOKEN=...
export HF_REPO_ID=<usuario>/sae-gpt2-small-l8-topk32
bash /workspace/setup_pod.sh
bash /workspace/minibeca/scripts/run_tmux.sh
tmux attach -t sae                     # ver progreso; Ctrl-b d para salir
```

El *watchdog* de `run_tmux.sh` relanza el entrenamiento con `--resume auto`
si cae; termina limpiamente cuando `final_inference/eval_metrics.json`
aparece.

### 2. Verificar y liberar

```bash
bash scripts/teardown.sh $POD_ID   # comprueba HF y borra el pod
```

### 3. Reproducción local (smoke test, sin GPU)

```bash
uv sync
make smoke
```

### 4. Análisis held-out (GPU recomendada; en CPU pica ~6 GB de RAM)

```bash
uv run python src/analyze_sae.py --stage metrics  --n-seqs 512 --ctx 128 --batch 32
uv run python src/analyze_sae.py --stage metrics  --n-seqs 128 --ctx 1024 --batch 8
uv run python src/analyze_sae.py --stage features --n-seqs 4096 --batch 64
uv run python src/analyze_sae.py --stage steer --latents 10274,19814,19443
```

### 5. Renderizar animaciones y portadas

```bash
make videos   # 6 escenas → assets/videos/
make covers   # portadas → assets/cover/
```

## 🧠 Conceptos clave

- **Polisemanticidad**: una misma neurona responde a conceptos distintos.
- **Superposición**: el modelo representa más *features* que dimensiones,
  proyectándolas sobre direcciones casi-ortogonales.
- **Sparse Autoencoder (TopK)**: diccionario sobrecompleto (`d_sae = 24 576`,
  `k = 32`) que descompone las activaciones residuales en *features*
  monosemánticas.
- **AuxK loss** (Gao et al. 2024): las latentes muertas se rescatan
  aprendiendo a reconstruir el residuo de las vivas.
