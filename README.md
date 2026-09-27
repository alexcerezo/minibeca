# Interpretando GPT-2 con un Sparse Autoencoder

> Entrenar un **Sparse Autoencoder** sobre `gpt2-small` y comprobar que sus
> latentes se corresponden con conceptos reconocibles — robo, peligro,
> meses del año, la firma de una agencia — y que la red los usa
> causalmente para generar texto.

- 📄 **Artículo** → [`docs/articulo/ARTICULO_X_v3.md`](docs/articulo/ARTICULO_X_v3.md)
- 🧠 **Pesos** → [`alexcerezo/sae-gpt2-small-l8-topk32`](https://huggingface.co/alexcerezo/sae-gpt2-small-l8-topk32) (HF Hub)
- 📈 **Curvas** → [W&B `minibeca-xmihura`](https://wandb.ai/alexcerezocontreras-university-of-malaga/minibeca-xmihura) — `rd8xu845` (con AuxK) · `sjdvx9oh` (sin AuxK)
- 🎬 **Animaciones** → [`assets/videos/`](assets/videos), [`assets/cover/`](assets/cover)

Financiado por las [**Minibecas XMihura**](https://x.com/XMihura). ⭐️ si te resulta útil.

---

## De qué va esto

Cuando le preguntas a un modelo de lenguaje por qué ha escrito algo,
responde con **aún más texto generado**, y nada garantiza que esa
respuesta describa lo que realmente produjo la salida. La alternativa es
inspeccionar el cálculo directamente. A eso se dedica la
**interpretabilidad mecanicista**: estudiar los mecanismos internos de
una red neuronal a partir de sus activaciones y sus pesos.

Este repositorio explica, desde cero y con código, una de sus
herramientas principales — el **Sparse Autoencoder (SAE)** — y lo que
encontré al entrenar uno sobre `gpt2-small`.

## El problema: neuronas polisemánticas

Dentro de cada MLP de GPT-2 hay 3 072 neuronas. La forma habitual de
interpretar una neurona es ver los textos en los que más se activa. Por
ejemplo, la **neurona 2787 del MLP del bloque 7** se activa fuertemente
en:

- `Up «in» heaven`
- `about 1«,»000 officers`
- `Eric H«ov»de`

Una preposición, el separador de miles y una sílaba de un apellido. Es
**polisemántica**: responde a varios conceptos sin relación. Aproximadamente
**dos de cada tres neuronas** de esta capa lo son.

La [**hipótesis de superposición**](https://transformer-circuits.pub/2022/toy_model/index.html) (Anthropic, 2022) lo explica: la red
representa **más características que dimensiones tiene su flujo residual**
(768 en GPT-2 small), asignando a cada una una dirección **casi
ortogonal**. Como las características son dispersas — en cada token solo
está presente una fracción mínima —, la interferencia es tolerable.
Consecuencia: cada eje (neurona) recibe contribuciones de muchas
características a la vez.

## La idea: buscar las direcciones, no los ejes

Si las características son **direcciones** del flujo residual, hay que
encontrarlas. Un **Sparse Autoencoder** es una red pequeña que aprende a
reconstruir su propia entrada a través de una representación intermedia
**mucho más grande y forzada a ser dispersa**:

- **Entrada**: flujo residual de un token, 768 dimensiones.
- **Encoder**: produce **24 576** activaciones (32× la entrada).
- **Regla TopK** ([Gao et al., OpenAI 2024](https://cdn.openai.com/papers/sparse-autoencoders.pdf)):
  se conservan **solo las 32 mayores** por token; el resto se anula.
- **Decoder**: reconstruye la entrada como suma ponderada de las
  **direcciones asociadas a esas 32 latentes**.
- **Diccionario**: el conjunto de las 24 576 direcciones.

Se entrena con 100 M tokens de OpenWebText minimizando el error de
reconstrucción. **Nadie le dice qué características buscar.** Si logra
reconstruir bien con solo 32 latentes por token, la hipótesis de
superposición predice que cada latente tenderá a alinearse con una
característica real de la red.

## Resultados

### 1. Reconstruye tan bien como los SAEs de referencia

Métricas en un **conjunto reservado** (held-out; texto no visto durante
el entrenamiento). El **CE loss score** mide qué fracción del daño que
causaría poner el flujo residual a cero (ablación) evita el SAE:

| SAE | Latentes | Activas/token | CE loss score |
|---|---:|---:|---:|
| **Este trabajo** | 24 576 | 32 | **98,4 %** |
| OpenAI (Gao et al.) | 32 768 | 32 | 98,4 % |
| Joseph Bloom (`res-jb`) | 24 576 | 67 (media) | 98,0 % |

### 2. Las latentes son interpretables; las neuronas no

Evaluación **ciega** (el evaluador no sabe si mira una latente o una
neurona) sobre 120 latentes vivas + 120 neuronas del MLP del bloque 7,
puntuadas de 1 a 5 por **dos LLMs independientes**:

- Correlación de Spearman entre evaluadores: **0,95**.
- Latentes: **4,34 / 5** — 87 % obtiene ≥ 4.
- Neuronas: **3,09 / 5** — 34 % obtiene ≥ 4.
- Probabilidad de la diferencia por azar (Mann-Whitney): ~10⁻²⁰.

Latentes con **5/5 de ambos evaluadores**, tal como salieron de la muestra aleatoria:

| # | Concepto |
|---|---|
| 10274 | robar (`steal`, `stole`, `stolen`, `theft`) |
| 19814 | la palabra `dangerous` |
| 17961 | nombres de mes dentro de una fecha (`On 17 March`) |
| 14748 | `Trek` y `Wars` inmediatamente después de `Star` |
| 11659 | `Reuters` como firma de agencia |
| 7840  | `played` en el sentido de "interpretar un papel" |
| 5183  | el `of` de `the role of` |
| 3433  | negación tras un "you" genérico (`You can't control…`) |

Hay características **semánticas** (robo, peligro), **sintácticas muy
específicas** (el `of` que sigue a `role`) y de **formato** (la firma de
una agencia). Ninguna se especificó de antemano.

### 3. Las latentes son causales, no solo correlacionales

**Steering**: mientras GPT-2 genera, se suma al flujo residual la
dirección de la latente escalada a su activación máxima observada. El
control es una **dirección aleatoria** de la misma magnitud.

Mismo comienzo, misma semilla, con y sin la latente **#10274**:

> **Sin intervención**: *…that after filing for bankruptcy, you will have to pay a penalty of up to $50,000…*
>
> **Con #10274**: *…that he stole nearly $50,000 from the taxpayers of the state…*

Con 80 continuaciones por condición y una lista de palabras del concepto **fijada antes** de ver los resultados:

| Latente | Sin intervención | Con la latente | Dirección aleatoria |
|---|---:|---:|---:|
| #10274 robar | 0 % | **32 %** | 3 % |
| #19814 peligro | 0 % | **32 %** | 0 % |
| #19443 color | 4 % | **55 %** | 0 % |

**Limitaciones**: no todas las latentes interpretables sirven para
dirigir la generación (de seis probadas, la de "beber" no superó al
control; la de "meses" se activó a sí misma sin que aparecieran más
meses). La intensidad tiene un óptimo: al doble de la activación máxima
el texto pierde coherencia.

### 4. Sin AuxK, el diccionario colapsa

Con TopK aparece un problema: **el gradiente solo llega a una latente
cuando queda entre las 32 elegidas**. Una latente que pierde muchas
veces seguidas deja de recibir gradiente y se muere.

En mi entrenamiento, en el **paso 1 190** de 12 207, **21 780 de las 24 576
latentes (89 %) estaban muertas**.

**Solución (AuxK, Gao et al.)**: entre las latentes muertas, se toman las
384 con mayor activación en ese token y se les pide que reconstruyan el
*residuo* — lo que las 32 vivas no explican —. Así vuelven a recibir
gradiente y aprenden justo lo que falta.

Ablación controlada — misma semilla, mismos datos, mismos
hiperparámetros, solo se elimina AuxK:

| | Con AuxK | Sin AuxK |
|---|---:|---:|
| Latentes que se activan ≥ 1 vez / 520 k tokens | **96,7 %** | 9,0 % |
| CE loss score | **98,4 %** | 96,3 % |

Sin AuxK, el SAE funciona con **~2 200 latentes útiles**; el resto de los
parámetros no contribuye a nada.

### 5. La longitud de contexto de entrenamiento importa

Entrené con textos de **1 024 tokens** — la ventana completa de GPT-2 —.
Los SAEs de referencia se entrenaron con 64 y 128 respectivamente,
porque se diseñaron para otros usos.

| SAE | Contexto de entrenamiento | CE loss score @ 1 024 tokens |
|---|---:|---:|
| **Este trabajo** | 1 024 | **98,3 %** |
| OpenAI | 64 | 90,3 % |
| Joseph Bloom | 128 | 88,2 % |

Además, el SAE de Bloom pasa de 67 latentes activas por token a **588** —
pierde la dispersión que se le pedía. La distribución del flujo residual
cambia con la posición en el texto; un SAE debe entrenarse con la
longitud de contexto con la que se va a usar.

### Error de medición evitado: el *attention sink*

La varianza explicada durante el entrenamiento salía 0,94; en held-out,
0,85. La diferencia venía del **primer token** de cada texto: GPT-2 lo
usa como **sumidero de atención**, y su flujo residual tiene una norma
unas **30× mayor** que la del resto. Al incluirlo, domina el cálculo y
hace que cualquier SAE parezca mejor de lo que es. **Todas las cifras de
este trabajo lo excluyen.**

## Uso

```python
from huggingface_hub import snapshot_download
from sae_lens import SAE

sae = SAE.load_from_disk(snapshot_download("alexcerezo/sae-gpt2-small-l8-topk32"))
z = sae.encode(x)      # activaciones de las 24 576 latentes para cada token
x_hat = sae.decode(z)  # reconstrucción del flujo residual
```

`x` es el flujo residual de `gpt2-small` en `blocks.8.hook_resid_pre`.
Consulta el artículo o [`src/analyze_sae.py`](src/analyze_sae.py) para el pipeline completo.

## Reproducir

```bash
uv sync

# Smoke test en CPU (~2 min): 20 pasos + resume bit-exacto
make smoke

# Análisis held-out (descarga los pesos de HF automáticamente)
uv run python src/analyze_sae.py --stage metrics  --n-seqs 512  --ctx 128  --batch 32
uv run python src/analyze_sae.py --stage metrics  --n-seqs 128  --ctx 1024 --batch 8
uv run python src/analyze_sae.py --stage features --n-seqs 4096 --batch 64
uv run python src/analyze_sae.py --stage steer    --latents 10274,19814,19443

# Renders del artículo
make videos    # 6 escenas Manim → assets/videos/
make covers    # 5 portadas + prisma Manim → assets/cover/
```

Entrenamiento completo en GPU alquilada (Runpod, ~26 min en A100 80 GB,
~0,70 USD): ver [`docs/PLAN_TECNICO.md`](docs/PLAN_TECNICO.md) y
[`scripts/`](scripts).

## Estructura

```
minibeca/
├── src/
│   ├── train_sae.py            # loop TopK + AuxK, checkpoints atómicos, --resume auto
│   ├── analyze_sae.py          # métricas held-out, features, steering causal
│   └── viz/                    # 6 escenas Manim + portadas
├── scripts/                    # ciclo GPU en Runpod (provisión → tmux watchdog → teardown)
├── data/
│   ├── analysis/               # salidas de analyze_sae.py (métricas, ratings, steering)
│   ├── wandb/                  # curvas exportadas (con y sin AuxK)
│   └── manim_data.json         # resumen que consumen las animaciones
├── assets/                     # productos finales (vídeos 1080p60, portadas)
├── docs/
│   ├── articulo/               # v3 (final), v2 (divulgativo), v1 (técnico)
│   ├── PLAN_TECNICO.md         # arquitectura, memoria, protocolo, infra
│   ├── ESTADO.md               # handoff
│   └── papers/                 # papers de referencia (HTML)
└── notebooks/                  # prueba de concepto en Colab
```

## Limitaciones

- Las puntuaciones de interpretabilidad son de **LLMs**, no de personas.
  La concordancia entre evaluadores es alta pero pueden compartir sesgos.
- Solo se evaluaron **activaciones máximas**. Una latente puede ser
  nítida en su top-k y difusa en las activaciones intermedias.
- **Un modelo, un punto de lectura**: GPT-2 small, `blocks.8.hook_resid_pre`.
  [Templeton et al. (Anthropic, 2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) mostraron que el método escala a
  Claude 3 Sonnet, pero no lo he reproducido.
- El **15 % de la varianza** queda sin explicar. El efecto sobre las
  predicciones es pequeño (CE score 98 %), pero es estructura que el
  diccionario no captura.

## Bibliografía

1. N. Elhage et al., [*Toy models of superposition*](https://transformer-circuits.pub/2022/toy_model/index.html) — Transformer Circuits Thread, 2022.
2. L. Gao et al., [*Scaling and evaluating sparse autoencoders*](https://cdn.openai.com/papers/sparse-autoencoders.pdf) — OpenAI, 2024.
3. T. Bricken et al., [*Towards monosemanticity: Decomposing language models with dictionary learning*](https://transformer-circuits.pub/2023/monosemantic-features/index.html) — Transformer Circuits Thread, 2023.
4. A. Templeton et al., [*Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet*](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) — Transformer Circuits Thread, 2024.
5. J. Bloom, [*SAELens*](https://github.com/jbloomAus/SAELens) — GitHub, 2024.

---

Este trabajo es resultado de las **[Minibecas XMihura](https://x.com/XMihura)**. Gracias a
[@XMihura](https://x.com/XMihura) por apostar por el talento joven y por
este proyecto en particular.
