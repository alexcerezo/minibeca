# Abrí GPT-2 y encontré su diccionario

### Un Sparse Autoencoder entrenado desde cero, contrastado con los de OpenAI y Joseph Bloom, puesto a prueba con evaluación ciega y con experimentos causales

---

Pregúntale a un modelo de lenguaje por qué ha dicho lo que ha dicho y te dará una explicación convincente. No hay forma de saber si es verdad. La única fuente fiable es lo que ocurre dentro: millones de números que se multiplican en cada token. Si queremos auditar estos sistemas antes de confiarles decisiones importantes, necesitamos leer esos números directamente.

El problema es que no se dejan leer. Abres GPT-2, eliges una neurona y miras los textos que más la activan:

> "Up« **in** » heaven", "about 1« **,** »000 officers", "Mor« **oc** »can security forces"

Una preposición, una coma de los miles y una sílaba de un gentilicio. La neurona no significa nada que podamos nombrar. Y no es un caso raro: en mi evaluación, dos de cada tres neuronas se comportan así.

Este artículo cuenta cómo entrené un **diccionario** que descompone el interior de GPT-2 en piezas que sí se entienden, y cómo lo puse a prueba para no engañarme. Esto es lo que encontré, con números, controles y todo el código abierto.

**Resumen en cuatro líneas:**

1. Mi diccionario reconstruye la capa 8 de GPT-2 igual de bien que el de OpenAI con un 25 % menos de piezas, y mejor que el de Joseph Bloom con la mitad de piezas activas por token.
2. En una evaluación a ciegas, el 87 % de sus piezas expresan un único concepto. Entre las neuronas del propio modelo, solo el 34 %.
3. Las piezas son palancas causales: sumar la de "color" hace que el 55 % de los textos generados hablen de colores, frente al 0 % con un empujón aleatorio de la misma fuerza.
4. Sin un truco concreto de entrenamiento (AuxK), el 91 % del diccionario muere y no vuelve. Lo comprobé repitiendo el entrenamiento entero sin él.

---

## Por qué las neuronas no se entienden

GPT-2 small representa cada token con un vector de 768 números. Pero el lenguaje tiene muchísimos más de 768 conceptos: nombres propios, tiempos verbales, temas, tonos, formatos. ¿Cómo caben?

La hipótesis que Anthropic formalizó en 2022 se llama **superposición**. El modelo no asigna una dimensión a cada concepto: los guarda en direcciones que están *casi* en ángulo recto entre sí. En 768 dimensiones caben muchísimas más direcciones casi perpendiculares que dimensiones. El precio es que se interfieren un poco, pero como en cada frase solo aparecen unos pocos conceptos a la vez, la interferencia apenas molesta.

> 🎬 **[Vídeo 1: SuperpositionScene]** *Cinco conceptos en dos dimensiones. Mientras solo uno esté activo, los demás apenas le molestan.*

Para el modelo es una compresión brillante. Para quien quiere entenderlo es un desastre: cada neurona es la suma de muchos conceptos a la vez, y por eso lo que la activa parece una lista aleatoria.

## La herramienta: un diccionario de 24 576 entradas

Un **Sparse Autoencoder** (SAE) intenta deshacer esa compresión. Toma el vector de 768 números y lo reescribe como una combinación de entradas de un diccionario mucho más grande, con una condición: en cada token solo puede usar **32 entradas de 24 576**. Luego intenta reconstruir el vector original a partir de esas 32.

Si lo consigue, cada entrada del diccionario (lo llamaré *latente*) tiene la oportunidad de representar un único concepto, porque ya no tiene que compartir espacio con los demás.

> 🎬 **[Vídeo 2: SAEAnatomyScene]** *768 números entran, solo 32 de 24 576 latentes se encienden, y a partir de ellas se reconstruye la entrada.*

Los detalles técnicos, para quien los quiera:

- **Dónde**: el flujo residual de GPT-2 small justo antes del bloque 8 (`blocks.8.hook_resid_pre`), el mismo punto donde existen SAEs públicos de referencia, así que se puede comparar directamente.
- **Arquitectura**: TopK (Gao et al., OpenAI 2024), con `k = 32` y expansión 32×.
- **Datos**: 100 millones de tokens de OpenWebText en secuencias de 1024 tokens.
- **Pérdida**: error de reconstrucción más el término auxiliar **AuxK**, que sale protagonista del primer hallazgo.

## Cómo evalué para no engañarme

Un SAE se puede presentar muy bien con las métricas que calcula el propio entrenamiento. Así que todo lo que sigue lo medí aparte, con un protocolo que fijé antes de mirar resultados:

- **Datos que el SAE nunca vio**: el último fragmento de OpenWebText, mientras que el entrenamiento solo leyó del primero.
- **Mismos tokens y mismo código para todos los SAEs**: el mío, el de OpenAI (TopK, 32 768 latentes) y el de Joseph Bloom (ReLU, 24 576 latentes), todos sobre el mismo punto de GPT-2. Bloom es un investigador de interpretabilidad mecanicista, creador de SAELens, la librería de código abierto más usada para entrenar y analizar SAEs. En 2024 publicó SAEs para todas las capas del flujo residual de GPT-2 small, que desde entonces sirven de referencia en el campo.
- **La métrica que importa**: sustituyo la capa 8 de GPT-2 por la reconstrucción del SAE y mido cuánto empeora el modelo prediciendo la siguiente palabra. Si el SAE no pierde nada, el modelo no nota la diferencia.

Un ejemplo de por qué esto importa. Durante el entrenamiento, el SAE reportaba una varianza explicada de 0,94. Con este protocolo sale **0,85**. La diferencia está en la primera posición de cada secuencia: GPT-2 la usa como "sumidero de atención" y su vector es unas 30 veces más grande que el resto. Si la incluyes, domina la varianza y hace que cualquier SAE parezca mejor de lo que es. Las cifras de este artículo la excluyen.

---

## Hallazgo 1: sin AuxK, el diccionario muere

Los SAE TopK tienen un problema conocido: las latentes que pierden la competición por estar entre las 32 elegidas dejan de recibir gradiente y se quedan muertas para siempre. En mi entrenamiento esto ocurrió de golpe. En el paso 1190, **21 780 de las 24 576 latentes (el 89 %) llevaban más de mil pasos sin activarse**.

El término AuxK pone a trabajar a esas latentes muertas: les pide que reconstruyan lo que las vivas no consiguen explicar. Durante los siguientes 3000 pasos fue el término dominante de la pérdida, y fue resucitando el diccionario hasta dejar **solo 42 latentes muertas** al final.

Esto hasta aquí es una correlación: AuxK se activó y las latentes revivieron. Para saber si fue AuxK quien las revivió, **repetí el entrenamiento completo, con los mismos datos, la misma semilla y los mismos hiperparámetros, pero sin AuxK**.

> 🎬 **[Vídeo 3: AuxKAblationScene]** *Dos entrenamientos idénticos salvo por un término de la pérdida.*

Sin AuxK, el colapso ocurre igual (22 768 latentes muertas en el paso 1310) y nunca se recupera: termina con 22 346. En texto nuevo:

| | Con AuxK | Sin AuxK |
|---|---:|---:|
| Latentes que llegan a activarse alguna vez (520 000 tokens) | **96,7 %** | 9,0 % |
| Pérdida extra del modelo al usar el SAE (nats) | **0,12** | 0,28 |
| CE loss score (1 = el modelo no nota nada) | **0,984** | 0,963 |
| Varianza explicada | **0,85** | 0,76 |

Sin AuxK, el SAE usa unas 2200 latentes y tira el resto, y el modelo pierde más del doble al pasar por él. **AuxK no es un ajuste fino: sin él, el 91 % del diccionario que entrenas no sirve para nada.**

## Hallazgo 2: las latentes se leen, las neuronas no

Que un SAE reconstruya bien no garantiza que sus piezas signifiquen algo. Para comprobarlo hice una evaluación ciega:

1. Elegí al azar **120 latentes vivas del SAE** y **120 neuronas del MLP de la capa 7** de GPT-2, que es la capa que escribe en el punto donde está el SAE.
2. Para cada una saqué los 12 textos que más la activan en datos nunca vistos.
3. Lo mezclé todo y se lo pasé a dos evaluadores automáticos independientes (dos modelos de lenguaje distintos) sin decirles qué era cada unidad. Tenían que puntuar de 1 (sin patrón) a 5 (un solo concepto nítido), con la escala del trabajo de Anthropic *Towards Monosemanticity*.

> 🎬 **[Vídeo 4: NeuronVsLatentScene]** *A la izquierda, una neurona de GPT-2. A la derecha, una latente del SAE. Después, las 240 puntuaciones.*

| | Neuronas MLP | Latentes SAE |
|---|---:|---:|
| Puntuación media | 3,09 | **4,34** |
| Puntuadas 4 o 5 | 34 % | **87 %** |
| Puntuadas 5 | 7 % | **50 %** |

Los dos evaluadores coincidieron casi del todo (correlación de Spearman 0,95), y la diferencia es abrumadora (Mann-Whitney, p ≈ 10⁻²⁰; diferencia de medias +1,25, con intervalo de confianza al 95 % de [1,03; 1,47]).

Algunas de las latentes que los dos evaluadores puntuaron con un 5, sacadas de la muestra aleatoria y no elegidas a dedo:

- **10274**: robar (*steal, stole, stolen, theft*)
- **19814**: la palabra *dangerous*
- **5183**: el *of* de "the role **of**"
- **17961**: meses dentro de fechas ("On 17 **March**")
- **7840**: *played* en el sentido de interpretar un papel
- **11659**: *Reuters* en créditos de agencia
- **3433**: la negación tras el "you" genérico ("You **can't** control…")
- **14748**: *Trek* y *Wars* después de *Star*

Hay de todo: conceptos semánticos (robo, peligro), construcciones gramaticales muy concretas (el *of* después de *role*) y convenciones de formato (créditos de agencia). Nadie le dijo al SAE que buscara nada de esto: lo encontró solo, por la presión de reconstruir con pocas piezas.

## Hallazgo 3: las latentes son palancas causales

Que una latente se active con textos sobre robos puede significar dos cosas: que *detecta* el concepto o que el modelo lo *usa* para decidir qué escribir. Para distinguirlas hay que intervenir.

El experimento consiste en sumar la dirección de una latente al flujo residual de GPT-2 mientras genera texto, con distintas intensidades. Después comparo con un **control**: una dirección aleatoria con exactamente la misma fuerza. Si lo que cambia el texto es solo el ruido, el control debería hacer lo mismo.

> 🎬 **[Vídeo 5: SteeringScene]** *El mismo prompt y la misma semilla, con y sin la latente de "robar".*

Un ejemplo real con el prompt *"The main thing to know is"*:

- **Sin intervención**: *"…that after filing for bankruptcy, you will have to pay a penalty of up to $50,000…"*
- **Con la latente 10274**: *"…that he **stole** nearly $50,000 from the taxpayers of the state…"*

La misma semilla de muestreo y la misma cifra, pero ahora es un robo.

Medido sobre 5 prompts × 16 muestras por punto, cuento qué porcentaje de textos mencionan el concepto. Las palabras clave las fijé antes de ver ningún resultado.

| Latente | Sin intervenir | Con la latente (mejor intensidad) | Control aleatorio |
|---|---:|---:|---:|
| "robo" (10274) | 0 % | **32 %** | 3 % |
| "peligro" (19814) | 0 % | **32 %** | 0 % |
| "color" (19443) | 4 % | **55 %** | 0 % |

Dos matices que no quiero esconder:

- **La curva sube y luego baja.** Si empujas demasiado (intensidad 2 o más), GPT-2 empieza a escribir incoherencias y el concepto desaparece con todo lo demás. Hay una zona útil y es estrecha.
- **No todas las latentes funcionan como palanca.** Probé seis. La de "beber" no movió nada por encima del control, la de "bromas" solo llegó al 15 %, y la de meses cambió la activación interna sin que aparecieran nombres de meses en el texto. Que una latente sea interpretable no garantiza que sea útil para dirigir al modelo.

## Hallazgo 4: entrenar con contexto largo importa

Los tres SAEs empatan cuando se evalúan con secuencias de 128 tokens (pérdida extra de 0,12 a 0,15 nats). Pero GPT-2 trabaja con contextos de hasta 1024 tokens, y ahí aparecen las diferencias:

> 🎬 **[Vídeo 6: ContextLengthScene]** *Los mismos tres SAEs, evaluados con 128 y con 1024 tokens.*

| | Contexto de entrenamiento | Pérdida extra a 128 | Pérdida extra a 1024 | CE score a 1024 |
|---|---:|---:|---:|---:|
| **Este trabajo** | 1024 | 0,12 | **0,14** | **0,983** |
| OpenAI TopK 32k | 64 | 0,12 | 0,81 | 0,903 |
| J. Bloom ReLU 24k | 128 | 0,15 | 0,99 | 0,882 |

El de Bloom pasa de 67 a **588 latentes activas por token** y su varianza explicada se vuelve negativa. El mío apenas se inmuta.

La explicación más probable es que las activaciones en posiciones avanzadas de la secuencia tienen una distribución distinta de las primeras, y un SAE que nunca las ha visto no sabe representarlas. No es un defecto de esos SAEs (se entrenaron para otro uso), pero es una advertencia práctica: **un SAE solo es fiable en la longitud de contexto con la que se entrenó**. Si vas a analizar documentos largos, entrénalo con documentos largos.

---

## Lo que esto no demuestra

- **Las puntuaciones de interpretabilidad son de modelos de lenguaje, no de humanos.** La concordancia entre dos evaluadores distintos es alta, pero ambos pueden compartir sesgos.
- **Los textos que más activan una latente solo muestran su pico.** Una latente puede ser nítida arriba y difusa en activaciones medias. Anthropic lo analiza con más detalle y yo no lo he hecho aquí.
- **Es un modelo pequeño y una sola capa.** GPT-2 small tiene 124 millones de parámetros. Anthropic mostró que la técnica escala hasta Claude 3 Sonnet, pero mis resultados concretos son de este modelo y esta capa.
- **El 15 % del comportamiento de la capa sigue sin explicar.** Con un 85 % de varianza explicada, hay estructura que el diccionario todavía no captura.

## Por qué importa

Si un modelo tiene una dirección interna para "peligro", podemos detectar cuándo la activa, y también empujarla o apagarla. Eso es exactamente lo que necesitamos para auditar modelos grandes: no preguntarles qué piensan, sino mirarlo.

Este trabajo no descubre nada que la literatura no hubiera anticipado. Lo que aporta es una réplica completa, con controles que suelen faltar (la ablación de AuxK, la evaluación ciega contra neuronas, el steering con control aleatorio, la comparación con SAEs públicos en los mismos tokens) y todo el material abierto para que cualquiera pueda repetirlo o rebatirlo.

## Todo es abierto

- **Código** (entrenamiento, análisis y estas animaciones): [github.com/alexcerezo/minibeca](https://github.com/alexcerezo/minibeca)
- **Pesos del SAE**: [huggingface.co/alexcerezo/sae-gpt2-small-l8-topk32](https://huggingface.co/alexcerezo/sae-gpt2-small-l8-topk32)
- **Curvas de entrenamiento** (con AuxK y sin AuxK): [proyecto de W&B](https://wandb.ai/alexcerezocontreras-university-of-malaga/minibeca-xmihura)

Cargarlo son tres líneas:

```python
from huggingface_hub import snapshot_download
from sae_lens import SAE
sae = SAE.load_from_disk(snapshot_download("alexcerezo/sae-gpt2-small-l8-topk32"))
```

Y `sae.encode(x)` te devuelve, para cada token, cuáles de las 24 576 latentes están encendidas.

---

*Este trabajo es mi entregable para las **Minibecas de XMihura**. Gracias a XMihura por apostar por que la investigación rigurosa en seguridad de IA también puede hacerse fuera de los grandes laboratorios.*

*Referencias: Elhage et al., "Toy Models of Superposition" (Anthropic, 2022) · Bricken et al., "Towards Monosemanticity" (Anthropic, 2023) · Templeton et al., "Scaling Monosemanticity" (Anthropic, 2024) · Gao et al., "Scaling and evaluating sparse autoencoders" (OpenAI, 2024).*
