# Plan técnico — SAE sobre `gpt2-small` (Minibecas XMihura)

Este documento cubre los bloques 1, 2, 3 y 5 del brief. Los bloques 4
(`src/train_sae.py`), 6 (`src/viz/manim_scenes.py`) y 7 (`docs/articulo/`) están en
sus propios ficheros.

---

## Bloque 1 — Arquitectura del SAE

### Comparativa L1 vs TopK vs Gated

|                             | **Standard L1** | **TopK** *(elegida)* | **Gated** |
|---|---|---|---|
| Restricción de escasez      | penalización `λ·‖z‖₁` | `‖z‖₀ = k` por construcción | encoder auxiliar de gating |
| Sensibilidad de HP          | *muy alta*: λ desplaza L₀/reconstrucción y hay que barrerlo por capa | ninguna — se fija `k`; no hay `λ` | media |
| Estabilidad matemática      | shrinkage sesga los latentes hacia 0; interacciona con la norma del decoder | recto: no hay shrinkage; el ranking top-k desacopla magnitud de activación | mejora el shrinkage con un coste extra en pesos |
| Coste de un backward pass   | 2·(d·m) matmul + `‖·‖₁` | 2·(d·m) matmul + `topk` (`O(m log k)`) | ~2× (gate + magnitud) |
| Latentes muertos            | frecuente; se mitiga con *ghost grads* o *resampling* | tratado nativamente por **AuxK** (Gao et al. 2024): las latentes muertas reconstruyen el residuo | comparable a TopK con L1 aux |
| Comparabilidad pública      | referencia clásica | referencia moderna (`gpt2-small-res-jb`, Neuronpedia, Anthropic 2024) | menos SAEs públicos |

**Decisión.** TopK. En un presupuesto de un único *run* (< 1 h, sin sweep de
hiperparámetros), fijar `L₀ = k` por construcción es lo único responsable: se
elimina el eje más tóxico del espacio (λ), el gradiente no sufre *shrinkage*,
y **AuxK** revive las latentes muertas sin *scaffolding* adicional. Además el
código de `TopKTrainingSAE` en SAELens 6.51 ya implementa la variante con
`rescale_acts_by_decoder_norm=True`, que evita el *drift* del norm del
decoder durante el entrenamiento.

### Capa objetivo

`blocks.8.hook_resid_pre` (equivalente a `blocks.7.hook_resid_post`), `d_in =
768`. Razones:

1. Zona intermedia-tardía donde emergen las *features* más abstractas
   (véase *Scaling Monosemanticity*, Anthropic 2024).
2. Existe un SAE público (`gpt2-small-res-jb`) exactamente en esta capa. Nos
   da una referencia de **explained variance ≈ 0.85** y **CE-loss score ≈
   0.95** contra la cual comparar de forma directa.
3. Antes de la capa 8 los residuos son demasiado tokénicos; después, la
   distribución empieza a colapsar hacia la predicción del siguiente token
   (menos útil interpretativamente).

### Dimensionalidad

Con hardware de centro de datos disponible durante ~1 h, el cuello ya no es
la VRAM sino los **tokens por latente**. Regla de Gao et al.: para que las
latentes se estabilicen hacen falta ≥ 3 000 tokens por latente activo
(`total_tokens · k / d_sae`). Con 100 M tokens y `k = 32`:

| Expansión | `d_sae` | Tokens/latente activo | Verdicto |
|---|---:|---:|---|
| 8×  | 6 144  | 520 000 | infrautilizado |
| 16× | 12 288 | 260 000 | óptimo tiempo/latente |
| **32×** | **24 576** | **130 000** | elegido (referencia pública) |
| 64× | 49 152 | 65 000 | riesgo de dead latents residuales |

**Decisión.** `expansion_factor = 32`, `d_sae = 24 576`, `k = 32`, `aux_loss_coefficient
= 1/32` (convención OpenAI/EleutherAI, no el `1.0` por defecto de SAELens).

---

## Bloque 2 — Auditoría de memoria y pipeline

Hardware objetivo: **A100 80 GB** o **L40S 48 GB**. Todo en fp32 para los
pesos, `bf16` autocast en el forward pass. Cifras conservadoras.

### Pesos y estados persistentes

| Componente | Parámetros | Bytes | MB |
|---|---:|---:|---:|
| `gpt2-small` bf16 (LN plegada, tied embeddings) | 124 M | 2 | 248 |
| SAE W_enc + W_dec (768 × 24 576 × 2)            | 37.7 M | 4 | 151 |
| SAE b_enc + b_dec                               | 25 k  | 4 | 0.1 |
| Adam m + v (fp32)                               | 2× SAE | 4 | 302 |
| **Subtotal persistente**                        |       |   | **≈ 701 MB** |

Adam en 8-bit ahorraría 226 MB → despreciable en 80 GB, y añade una dependencia
más y un ligero coste de fidelidad. **No lo usamos.**

### Activaciones transitorias por paso (`batch = 8192`, `d_in = 768`, `d_sae = 24 576`)

| Tensor | Forma | Bytes fp32 | MB |
|---|---|---:|---:|
| `sae_in` (batch de activaciones)   | 8 192 × 768    | 4 | 25   |
| `hidden_pre = x W_enc + b_enc`     | 8 192 × 24 576 | 4 | 805  |
| `feature_acts` (dispersas, k=32)   | 8 192 × 24 576 | 4 | 805  |
| `sae_out`                          | 8 192 × 768    | 4 | 25   |
| Máscara TopK + índices             | ~ 8 192 × 24 576 | 1+8 | ~90 |
| Gradientes de `hidden_pre`/`feats` | 2× arriba      | 4 | ~1 600 |
| **Pico activaciones/paso**         |                |   | **≈ 3.3 GB** |

### Buffer de activaciones del store (`n_batches_in_buffer = 32`)

`store_batch_size_prompts × context_size` = 32 × 1024 = 32 768 tokens por
recarga. Buffer total = `n_batches_in_buffer × 32 768 × d_in × 4 B` = **3.2 GB**
en fp32 (residente en GPU, half-refill según `mixing_buffer`).

### Forward LM (`stop_at_layer = 9`, `run_with_cache`)

Cache reducida al hook único: `32 × 1024 × 768 × 2 B` = **48 MB**. Los
tensores intermedios de atención suben transitoriamente ~600 MB pero se
liberan tras el hook.

### Suma total y margen

`701 + 3 300 + 3 200 + 700 (CUDA ctx + cuBLAS) ≈ 8 GB` de pico ⇒ **12–14 %
de un A100 80 GB, 17–18 % de un L40S 48 GB**. Hay margen 4–5× para subir
`expansion_factor` a 64 o `context_size` a 2048 si un experimento posterior
lo requiere.

### Configuración concreta que se usa

```
context_size            = 1024
store_batch_size_prompts= 32
n_batches_in_buffer     = 32
train_batch_size_tokens = 8192
prepend_bos             = True
exclude_bos             = True   # excluir BOS: norma outlier
precision               = bf16   # autocast en LM y en SAE forward
weights dtype           = fp32   # pesos SAE, Adam m/v
normalize_activations   = "none" + escalador constante externo
                                     (compatible con TopK aux-loss)
```

---

## Bloque 3 — Protocolo de entrenamiento

### Hiperparámetros

| Parámetro | Valor | Justificación |
|---|---|---|
| Optimizador | Adam(β₁=0.9, β₂=0.999, ε=6.25·10⁻¹⁰) | OpenAI: ε ≈ 1/(batch·d_in) |
| LR base    | 3·10⁻⁴ | rango estable para TopK con `decoder_init_norm=0.1` |
| Warmup     | 1 000 pasos lineales | evita explosión de la aux-loss en los primeros pasos |
| Schedule   | plateau → **cosine** en el último 20 % hasta `lr/10` | fine-tune blando; ver `build_scheduler` |
| Grad clip  | ‖g‖ ≤ 1.0 | estabiliza picos por batches raros |
| `k`        | 32     | ver bloque 1 |
| `aux_loss_coefficient` | 1/32 | AuxK con convención OpenAI |
| `decoder_init_norm`    | 0.1  | heurística Anthropic abril 2024 |
| Escalado activaciones  | `α = √d_in / mean‖x‖` constante | equivalente a `expected_average_only_in` pero fuera del store; se pliega al final con `fold_activation_norm_scaling_factor` |
| `dead_feature_window`  | 1 000 pasos (≈ 8 M tokens) | umbral para marcar dead latents en la máscara AuxK |

Cálculo de pasos: 100 M tokens / 8 192 = **12 207 pasos**. Warmup = 8 %,
decay = 20 %, plateau = 72 %.

### Detección y manejo de dead features

Contador por latente `n_forward_passes_since_fired`, reset a 0 cuando la
latente dispara al menos un token del batch. Máscara `dead_mask =
counter > 1000` alimenta directamente el término AuxK (`k_aux = d_sae / 2`,
reconstruye el residuo con solo las latentes muertas). No usamos
*resampling* (opcional en SAELens): AuxK es suficiente y no rompe la
reproducibilidad.

Además, `act_freq_scores` acumula frecuencia de disparo y se persiste en el
checkpoint para trazar `sparsity/log10_density_mean` en W&B (histograma de
Bloom).

### Métricas en Weights & Biases

Cada `wandb_log_every = 10` pasos:

- `loss/total`, `loss/mse`, `loss/aux`
- `sparsity/dead_features`, `sparsity/log10_density_mean`
- `opt/lr`, `throughput/tokens_seen`, `throughput/tokens_per_sec`

Cada `eval_every_steps = 500` (y al final) se corre `sae_lens.run_evals` con
`EvalConfig(compute_ce_loss=True, compute_kl=True, compute_l2_norms=True,
compute_sparsity_metrics=True, compute_variance_metrics=True)`, que reporta:

- `ce_loss_with_sae`, `ce_loss_without_sae`, `ce_loss_with_ablation`,
  `ce_loss_score = (ablation − with) / (ablation − without)`
- `explained_variance`, `mse`, `cossim`
- `l0` (efectivo, deberá ser ≈ 32)
- `kl_div_with_sae`

Objetivo publicable (referencia de `gpt2-small-res-jb` layer 7):
`explained_variance ≥ 0.80`, `ce_loss_score ≥ 0.90`, `l0 ≈ 32`,
`dead_features / d_sae < 0.05`.

---

## Bloque 5 — Infraestructura y resiliencia

Todo el ciclo vive en un pod GPU con un Network Volume montado en
`/workspace`, operado por SSH y `tmux`. Los caches (`HF_HOME`,
`WANDB_DIR`, `checkpoints`) van al volumen ⇒ sobreviven al pod.

### Ciclo (scripts en `scripts/`)

1. `provision_pod.sh` — crea (o reutiliza) el volumen y arranca el pod con:
   - `--template-id runpod-torch-v280` (torch 2.8.0 + CUDA 12.8, Python 3.12
     de sistema; **PEP 668**: instalar con `pip --break-system-packages`),
   - `--gpu-id "NVIDIA A100 80GB PCIe"` (o `NVIDIA L40S`),
   - `--network-volume-id … --volume-mount-path /workspace`,
   - `--ssh --terminate-after +3h` (*cost guard*: el pod se borra solo).
   Imprime `POD_ID`, `POD_IP`, `POD_PORT`, `POD_KEY`.
2. `setup_pod.sh` — dentro del pod: `pip --break-system-packages` para
   `sae-lens`, `transformer-lens`, `wandb`, `huggingface_hub`, `datasets`,
   `safetensors`; exporta `HF_HOME=/workspace/hf-cache` y `WANDB_DIR=/workspace/wandb`;
   `wandb login` y `HfFolder.save_token` no-interactivos si las env-vars
   están; clona el repo en `/workspace/minibeca`.
3. `run_tmux.sh` — arranca una sesión `tmux` con tres ventanas:
   - **train**: bucle *watchdog* que ejecuta `python src/train_sae.py
     --resume auto`; si el proceso muere, hace *backoff* 30 s y reintenta;
     sale limpio cuando aparece `final_inference/eval_metrics.json`.
   - **gpu**: `nvidia-smi --query-gpu=...` cada 15 s a
     `/workspace/logs/gpu.csv` (timestamp, util, mem, temp, potencia).
   - **log**: `tail -F /workspace/logs/train.log`.
4. `teardown.sh` — verifica que `sae_weights.safetensors` está en el repo
   HF y borra el pod. El volumen persiste hasta que se elimine explícitamente.

### Resiliencia frente a caídas

- **Watchdog en tmux**: cualquier crash del proceso (OOM transitorio,
  desconexión de red, etc.) provoca un reintento automático con
  `--resume auto`, que localiza el último checkpoint íntegro (verificado por
  SHA-256 en `manifest.json`) y restaura pesos + optimizador + scheduler +
  RNG (`torch`, `cuda`, `numpy`, `random`) + estado del `ActivationsStore` +
  `wandb_run_id` para no romper la curva.
- **Checkpoint atómico**: escritura a `step_XXXXXXXXXX.tmp/`, `fsync`
  recursivo, `os.replace` para publicar, `manifest.json` se escribe último
  como *sentinel* de completitud. Un apagado en medio del volcado deja el
  `.tmp/` corrupto ⇒ se ignora en el resume; el checkpoint anterior sigue
  siendo válido.
- **SIGTERM/SIGINT**: `GracefulKiller` en `train_sae.py` fuerza un
  checkpoint al final del paso actual antes de salir; el reinicio por
  Runpod (pull de la instancia) por lo tanto no pierde progreso.
- **Deadline blando**: `--deadline-hours 0.9` fuerza guardar y salir antes
  de que el `--terminate-after +3h` del pod entre en juego.
- **Telemetría persistente**: `gpu.csv` y `train.log` viven en el volumen;
  al inspeccionar post-mortem, ambos siguen disponibles aunque el pod ya
  no exista.

---

## Resultados reales (run `pleasant-paper-1`, 2026-09-26)

Corrida ejecutada según este plan. Métricas del snapshot de evaluación
`@ step 12000` sobre 131 072 tokens held-out para reconstrucción y 65 536
para sparsity/varianza:

| Métrica | Valor | Notas |
|---|---:|---|
| `sparsity.l0` | **32.0** | por construcción TopK |
| `sparsity.l1` | 82.45 | |
| `reconstruction_quality.explained_variance` | **0.9413** | |
| `reconstruction_quality.explained_variance_legacy` | 0.8615 | denominador global (peor caso) |
| `reconstruction_quality.mse` | 923.0 | |
| `reconstruction_quality.cossim` | 0.9553 | |
| `model_performance_preservation.ce_loss_score` | **0.9812** | (abl − sae)/(abl − clean) |
| `model_performance_preservation.ce_loss_with_sae` | 3.266 nats | |
| `model_performance_preservation.ce_loss_without_sae` | 3.109 nats | LM intacto |
| `model_performance_preservation.ce_loss_with_ablation` | 11.438 nats | activación puesta a 0 |
| `model_behavior_preservation.kl_div_score` | **0.9812** | |
| `model_behavior_preservation.kl_div_with_sae` | 0.155 nats | |
| `shrinkage.l2_norm_in` | 105.0 | activación real, promedio |
| `shrinkage.l2_norm_out` | 100.3 | reconstrucción, promedio |
| `shrinkage.l2_ratio` | 0.9545 | 4.5 % de shrinkage — clásico de TopK |
| `shrinkage.relative_reconstruction_bias` | 1.0001 | sesgo escalar, no direccional |
| dead features (final) | **42 / 24 576 (0.17 %)** | |
| wall time | 26 min | |
| tokens | 100 040 448 | |
| throughput medio | 68 089 tok/s | GPU al 98-99 % util |
| coste | 0.69 $ | |

### Hallazgos empíricos

1. **Resurrección de latentes muertos.** En el step 1 190 el TopK ha
   seleccionado sus favoritos y **21 780 latentes (89 %)** llevan más de
   1 000 pasos sin dispararse. La AuxK loss se dispara a **3.89** en la
   ventana step 1 200-1 300 y en 4 000 pasos rescata a **21 738 de ellas**;
   sólo 42 quedan muertas al final.
2. **Meseta temprana.** Las métricas de evaluación llegan a su asíntota
   alrededor del step 4 000. De 4 000 a 12 200 la explained variance
   sube de 0.931 → 0.941 (+1 pt). El **93 %** de la mejora ocurre en el
   primer **33 %** del compute. Presupuestar por meseta y no por
   token-target absoluto.
3. **AuxK como evento, no como fase.** La aux-loss se mantiene en cero
   1 200 pasos, sube en 80, y decae en 3 500. Un schedule que la active
   sólo al detectar el primer dead-latent ahorraría cómputo sin cambiar
   el resultado.
4. **La CE loss casi no se degrada.** `ce_loss_with_sae − without =
   0.157 nats` (5 % de degradación relativa). La caja de la ablación a
   cero cuesta **8.33 nats** más — el SAE captura el 98 % del uplift.
5. **Shrinkage escalar y estable.** `l2_ratio = 0.955` con
   `relative_reconstruction_bias ≈ 1.000`: sesgo escalar plano, no
   direccional. Compensable con un único factor si un downstream lo
   necesita.
6. **VRAM sobrada — el cuello es data pipeline.** GPU al 98-99 % util
   con solo 12 GB / 80 GB usados; peak durante `run_evals` sube a 28 GB
   por el batched forward. Escalar `d_sae` a 65 536 costaría ~30 % más
   tiempo, no más memoria.
