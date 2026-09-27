# Estado del proyecto — Minibeca XMihura

**Fecha:** 2026-09-26.  **Estado:** 7 bloques entregados y verificados.

## Entregables

| Bloque | Fichero | Estado |
|---|---|---|
| 1. Arquitectura SAE + comparativa | `docs/PLAN_TECNICO.md` §Bloque 1 | ✅ |
| 2. Auditoría de memoria | `docs/PLAN_TECNICO.md` §Bloque 2 | ✅ |
| 3. Protocolo de entrenamiento | `docs/PLAN_TECNICO.md` §Bloque 3 | ✅ |
| 4. Script de producción con tolerancia a fallos | `src/train_sae.py` | ✅ |
| 5. Infraestructura y resiliencia | `docs/PLAN_TECNICO.md` §Bloque 5 + `scripts/*.sh` | ✅ |
| 6. Animaciones Manim (6 escenas, datos leídos de `data/`) | `src/viz/manim_scenes.py` → `assets/videos/` | ✅ |
| 7. Artículo para X | `docs/articulo/ARTICULO_X_v2.md` (v1 técnica: `ARTICULO_X_v1.md`) | ✅ |
| 8. Verificación independiente held-out + ablación AuxK + evaluación ciega + steering | `src/analyze_sae.py`, `data/analysis/` | ✅ |

## Verificaciones ejecutadas

- **Python `py_compile`** de `src/train_sae.py` y `src/viz/manim_scenes.py` → OK.
- **`bash -n`** de los cuatro scripts en `scripts/` → OK.
- **Smoke test end-to-end en CPU** (`--smoke`, gpt2 real, wikitext-103):
  - run completo hasta step 20 con `final_inference/eval_metrics.json` presente.
  - kill mid-run + `--resume auto`: fast-forward del dataset, restauración de
    optimizador + scheduler + RNG + estado del store, continuación bit-exacta.
  - corrupción intencional del checkpoint más reciente: `manifest.json` con
    SHA-256 lo detecta y el resume cae al checkpoint anterior íntegro.
- **Render Manim** de las 6 escenas (`SuperpositionScene`, `SAEAnatomyScene`,
  `AuxKAblationScene`, `NeuronVsLatentScene`, `SteeringScene`,
  `ContextLengthScene`) con QA visual de fotogramas clave.
- **Análisis held-out** (GPU alquilada, borrada al terminar): métricas ctx 128
  y 1024 contra OpenAI y Bloom, densidad sobre 520 k tokens, ablación completa
  sin AuxK (W&B `sjdvx9oh`, pesos en `outputs/sae_noauxk/`), evaluación ciega de
  120 latentes y 120 neuronas (`interp_ratings.json`), steering con control
  aleatorio (`steering.json`).

## Configuración fija (arquitectura)

- Modelo: `gpt2-small`, hook `blocks.8.hook_resid_pre` (≡ `blocks.7.hook_resid_post`).
- SAE: **TopK, k=32, expansión 32× → d_sae = 24 576**, AuxK con `α = 1/32`,
  `rescale_acts_by_decoder_norm=True`, `decoder_init_norm=0.1`.
- Datos: `apollo-research/Skylion007-openwebtext-tokenizer-gpt2` streaming,
  100 M tokens, `context_size=1024`, `train_batch_size_tokens=8192`,
  `store_batch_size_prompts=32`, `n_batches_in_buffer=32`, BOS excluido.
- Escalado: constante `α = √d_in / mean‖x‖` estimado sobre 100 batches y
  plegado en los pesos con `fold_activation_norm_scaling_factor` al final.
- Optim: Adam(β=(0.9, 0.999), ε=6.25·10⁻¹⁰), lr=3·10⁻⁴, warmup 1000 pasos,
  cosine en el último 20 % hasta `lr/10`, grad clip 1.0.
- Checkpoints: `/workspace/checkpoints/sae-gpt2/`, atómicos con manifest
  SHA-256, últimos 3 conservados, symlink `latest`, cada 1000 pasos o
  600 s (lo que llegue antes) + al recibir SIGTERM.
- Eval: `sae_lens.run_evals` cada 500 pasos (CE-loss, KL, L₂, sparsity,
  varianza) + evaluación final antes del upload.

## Configuración fija (infra de despliegue)

- Template: `runpod-torch-v280` (torch 2.8.0 + CUDA 12.8 + Python 3.12 +
  Ubuntu 24.04). Nota PEP 668: `pip install --break-system-packages`.
- GPU: A100 80GB PCIe (o L40S 48GB como alternativa).
- Disco: container 20-30 GB efímero + **Network Volume 50 GB** en
  `/workspace` en el mismo DC que el pod.
- Cost guard: `--terminate-after +3h` (run real ≈ 40-60 min).
- Autenticación (env vars antes del `ssh`): `RUNPOD_API_KEY`, `HF_TOKEN`,
  `WANDB_API_KEY`, `HF_REPO_ID`.

## Cómo lanzar

```bash
# Local (una vez, para verificar)
uv sync
make smoke
make videos   # 6 escenas → assets/videos/

# Producción
bash scripts/provision_pod.sh
scp -i $POD_KEY -P $POD_PORT scripts/*.sh root@$POD_IP:/workspace/
ssh -i $POD_KEY -p $POD_PORT root@$POD_IP \
    'WANDB_API_KEY=... HF_TOKEN=... HF_REPO_ID=... \
     bash /workspace/setup_pod.sh && bash /workspace/minibeca/scripts/run_tmux.sh'
tmux attach -t sae   # opcional, ver la curva
bash scripts/teardown.sh $POD_ID
```

## Pendiente del usuario (para autonomía plena mía en Runpod)

1. Crear `RUNPOD_API_KEY` en https://console.runpod.io/user/settings.
2. (Opcional) `HF_TOKEN` con permiso write + `WANDB_API_KEY`.
3. Pegármelas en el chat → instalo `runpodctl`, hago `runpodctl config`,
   y ejecuto el ciclo completo.

Sin la key: el usuario ejecuta los tres scripts manualmente; el resultado
funcional es idéntico.
