---
layout: default
title: Interpretando GPT-2 con un Sparse Autoencoder
description: Sparse Autoencoder TopK sobre gpt2-small; reconstrucción held-out, interpretabilidad ciega y steering causal.
---

<script>
window.MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] }, svg: { fontCache: 'global' } };
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<style>
  .page-header { background: #111; background-image: none; padding: 3rem 1rem; }
  .project-name { font-weight: 600; }
  .project-tagline { opacity: 0.85; }
  .main-content { max-width: 68ch; font-size: 17px; line-height: 1.65; }
  .main-content h1, .main-content h2, .main-content h3 { color: #111; border: none; margin-top: 2.5rem; }
  .main-content h2 { font-size: 1.5em; }
  .main-content h3 { font-size: 1.2em; }
  .main-content a { color: #0b7285; }
  .main-content code { background: #f4f4f4; padding: 0.1em 0.35em; border-radius: 3px; font-size: 0.9em; }
  .main-content pre { background: #0f172a; color: #e2e8f0; padding: 1rem; border-radius: 6px; overflow-x: auto; }
  .main-content pre code { background: transparent; color: inherit; padding: 0; }
  .main-content blockquote { border-left: 3px solid #0b7285; color: #333; }
  .main-content table { display: block; overflow-x: auto; font-size: 0.95em; }
  .main-content th, .main-content td { padding: 0.4rem 0.7rem; border: 1px solid #e5e7eb; }
  .main-content th { background: #f9fafb; }
  video { max-width: 100%; height: auto; display: block; margin: 1.5rem auto; border-radius: 6px; box-shadow: 0 2px 12px rgba(0,0,0,0.15); background: #000; }
  figure { margin: 2rem 0; }
  figcaption { font-size: 0.9em; color: #666; text-align: center; margin-top: 0.5rem; font-style: italic; }
  .meta { margin: 0 0 1.5rem 0; color: #555; font-size: 0.95em; }
  .meta a { margin-right: 1em; }
</style>

<p class="meta">
  <a href="https://github.com/alexcerezo/minibeca">Código</a>
  <a href="https://huggingface.co/alexcerezo/sae-gpt2-small-l8-topk32">Pesos (HF)</a>
  <a href="https://wandb.ai/alexcerezocontreras-university-of-malaga/minibeca-xmihura">Curvas (W&amp;B)</a>
</p>

Cuando le preguntas a un modelo de lenguaje por qué ha escrito algo,
responde con aún más texto generado y nada garantiza que esa respuesta
describa lo que realmente produjo la salida.

La alternativa es inspeccionar ese cálculo directamente. A eso se dedica
la **interpretabilidad mecanicista**, el estudio de los mecanismos
internos de una red neuronal a partir de sus activaciones y sus pesos.
Este artículo explica, desde cero, una de sus herramientas principales,
el **Sparse Autoencoder**, y lo que encontré al entrenar uno sobre GPT-2.

## GPT-2 por dentro

GPT-2 es un modelo de lenguaje que se basa en una red neuronal entrenada
para predecir el siguiente fragmento de un texto a partir de los
anteriores. Para generar texto, predice un fragmento, lo añade al texto
y repite.

Una **red neuronal** es una función matemática compuesta por muchas
operaciones sencillas encadenadas, principalmente multiplicaciones de
matrices y funciones no lineales. Su comportamiento depende de sus
**parámetros** o **pesos**, los números de esas matrices, que se ajustan
durante el entrenamiento. Un modelo como GPT-2 small, la versión que
usé, tiene 124 millones de parámetros.

El texto entra en la red dividido en **tokens**, las unidades mínimas
con las que trabaja el modelo. Pueden ser palabras completas
(`" dangerous"`), fragmentos de palabras (`"Mor"`, `"occan"`) o signos
(`","`). GPT-2 tiene un vocabulario de 50 257 tokens y su ventana de
contexto abarca hasta 1 024 seguidos.

Cada token se convierte en un vector de 768 componentes, que se puede
interpretar como un punto — o una flecha — en un espacio de 768
dimensiones. Ese vector es la representación interna del token.

La red está organizada en **12 bloques** consecutivos. Cada bloque
contiene dos piezas:

- Una capa de **atención**, que permite a cada token incorporar
  información de los tokens anteriores del texto.
- Un **perceptrón multicapa** (MLP), que transforma la representación de
  cada token por separado.

Cada bloque no sustituye el vector del token, sino que le **suma** su
resultado. Por eso el vector que recorre la red se llama **flujo
residual** (*residual stream*): cada bloque lee de él y escribe en él.
Y el valor que toma en un punto concreto de la red, para un texto
concreto, es una **activación**.

Todo el análisis se hace sobre el flujo residual **tras los primeros 8
de los 12 bloques**. Elegí ese punto por dos motivos. Está a dos tercios
del recorrido, donde la red ya ha tenido ocho bloques para construir
representaciones abstractas, pero todavía le quedan cuatro para
convertirlas en una predicción. Y es el mismo punto en el que están
entrenados los SAEs públicos de referencia, así que puedo comparar
resultados en igualdad de condiciones.

## El problema: neuronas polisemánticas y superposición

Dentro de cada MLP hay 3 072 **neuronas**. Una neurona es una unidad que
calcula un único número a partir del flujo residual. Cuanto mayor es ese
número, se dice que la neurona está más **activada**.

La forma habitual de interpretar una neurona es buscar los textos en los
que más se activa. Por ejemplo, estos son los tokens que más activan la
**neurona 2787 del MLP del bloque 7**, marcados entre `« »`:

> `Up «in» heaven` · `about 1«,»000 officers` · `Eric H«ov»de`

Una preposición, el separador de miles y una sílaba de un apellido. Una
neurona que responde a varios conceptos sin relación entre sí se llama
**polisemántica**. En mi evaluación, aproximadamente **dos de cada tres
neuronas** lo eran.

La explicación con más apoyo empírico es la **hipótesis de superposición**
enunciada en *Toy Models of Superposition* [1]. Para enunciarla hace
falta un término más: una **característica** (*feature*) es una propiedad
del texto que la red representa internamente, como "este token es un
mes" o "el texto trata de un robo".

La hipótesis dice que la red representa **más características que
dimensiones tiene su flujo residual**, y que lo hace asignando a cada
característica una **dirección**, un vector concreto del espacio de 768
dimensiones. Cuando la característica está presente, el flujo residual
se desplaza en esa dirección.

En 768 dimensiones solo caben 768 direcciones perfectamente
**ortogonales** (perpendiculares entre sí, de modo que no interfieren).
Pero caben muchísimas más direcciones **casi ortogonales**, que
interfieren relativamente poco entre sí. La red las aprovecha, de manera
similar a como se distribuyen los electrones en un átomo. La
interferencia es tolerable porque las características son **dispersas**
(*sparse*): en cada token solo está presente una fracción mínima de
todas ellas.

La consecuencia para la interpretabilidad es directa. Si las
características son direcciones que **no coinciden con los ejes**, cada
neurona (cada eje) recibe contribuciones de muchas características a la
vez. Por eso las neuronas son polisemánticas.

<figure>
  <video src="assets/videos/SuperpositionScene.mp4" controls muted playsinline preload="metadata"></video>
  <figcaption>Cinco características representadas en un espacio de dos dimensiones. Mientras solo una está activa, las demás apenas interfieren.</figcaption>
</figure>

## El Sparse Autoencoder

Si las características son direcciones del flujo residual, el objetivo
es encontrarlas. Un **Sparse Autoencoder** (SAE, *autoencoder disperso*)
es una red neuronal pequeña que se entrena para eso.

Un **autoencoder** es una red que aprende a reproducir su propia
entrada. Tiene dos partes:

- El **encoder**, que transforma la entrada en una representación
  intermedia.
- El **decoder**, que intenta reconstruir la entrada a partir de esa
  representación.

En un SAE, la representación intermedia es mucho más grande que la
entrada, pero se le obliga a ser **dispersa**: casi todos sus valores
deben ser cero. En mi caso:

- La **entrada** es el flujo residual de un token, un vector de 768
  dimensiones.
- El **codificador** produce 24 576 números, 32 veces más que la
  entrada. Cada uno es la activación de una **latente**.
- Cada latente tiene asociada una **dirección** en el espacio de 768
  dimensiones. El conjunto de las 24 576 direcciones es el
  **diccionario**.
- El **decodificador** reconstruye la entrada como una suma ponderada de
  direcciones del diccionario, donde $z_i$ es la activación de la
  latente $i$ y $b$ es un vector fijo de sesgo:

$$
\hat{x} = b + \sum_i z_i\, d_i
$$

La dispersión se impone con la regla **TopK**, definida en *Scaling and
evaluating sparse autoencoders* [2]. En cada token se conservan las
$k = 32$ latentes con mayor activación y las otras 24 544 se ponen a
cero. Por tanto, cada flujo residual se describe como la combinación de
**solo 32 direcciones** del diccionario.

La hipótesis de superposición predice que, si el SAE logra reconstruir
bien con tan pocas latentes, cada latente tenderá a **alinearse con una
característica real** de la red. Aunque esto es algo que hay que
comprobar.

<figure>
  <video src="assets/videos/SAEAnatomyScene.mp4" controls muted playsinline preload="metadata"></video>
  <figcaption>Entra un vector de 768 dimensiones; se activan 32 de las 24 576 latentes y se reconstruye el vector.</figcaption>
</figure>

## Cómo se entrena

Entrenar una red es ajustar sus parámetros para minimizar una **función
de pérdida** que mide cuánto se equivoca. La pérdida principal del SAE
es el **error cuadrático de reconstrucción**, la distancia al cuadrado
entre el flujo residual original y su reconstrucción:

$$
\mathrm{Loss} = \|x - \hat{x}\|^2
$$

El ajuste se hace por **descenso de gradiente**: se calcula en qué
dirección hay que modificar cada parámetro para que la pérdida baje y se
le aplica un paso pequeño en esa dirección. Se repite con muchos
ejemplos.

Nadie le indica al SAE qué características buscar. Solo recibe flujos
residuales de GPT-2 procesando **100 millones de tokens** de OpenWebText
(una colección de páginas web) y el objetivo de reconstruirlos con 32
latentes por token.

## Cómo medir si funciona

Un error cuadrático bajo no basta. El SAE podría reconstruir bien la
mayor parte del vector y perder precisamente la información que GPT-2
usa para predecir. Así que medí el efecto **sobre la tarea del modelo**.

La calidad de las predicciones de un modelo de lenguaje se mide con la
**entropía cruzada** (*cross-entropy loss*): el promedio, sobre todos
los tokens, del logaritmo negativo de la probabilidad que el modelo
asignó al token que realmente venía a continuación. Si acierta con
seguridad, esa probabilidad se acerca a 1 y la entropía cruzada a 0. Se
expresa en **nats** (la unidad que resulta de usar el logaritmo
natural).

El procedimiento:

1. Mido la entropía cruzada de **GPT-2 sin modificar**: **3,59 nats**.
2. En el punto de lectura, **sustituyo** el flujo residual por la
   reconstrucción del SAE y dejo que el resto de la red continúe.
   Resultado: **3,71 nats**.
3. Como referencia, sustituyo el flujo residual por ceros
   (**ablación**): **11,11 nats**.

El **CE loss score** resume estas tres cifras: qué fracción del daño
causado por la ablación evita el SAE.

$$
\text{CE score} = \frac{11{,}11 - 3{,}71}{11{,}11 - 3{,}59} = 98{,}4\ \%
$$

Todas las métricas se calcularon sobre un **conjunto reservado**
(*held-out*): texto que el SAE no vio durante el entrenamiento, para
medir si **generaliza** y no solo si memoriza. Con los mismos textos y
el mismo código evalué dos SAEs públicos entrenados en el mismo punto de
GPT-2.

| SAE | Latentes | Activas por token | CE loss score |
|---|---:|---:|---:|
| **Este trabajo** | 24 576 | 32 | **98,4 %** |
| OpenAI (Gao et al.) | 32 768 | 32 | 98,4 % |
| Joseph Bloom | 24 576 | 67 (media) | 98,0 % |

## Sin AuxK, el diccionario colapsa

Con la regla TopK aparece un problema. El **gradiente solo llega a una
latente cuando esta queda entre las 32 elegidas**. Una latente que
pierde muchas veces seguidas deja de recibir gradiente, no mejora y
sigue perdiendo. Cuando una latente no se activa en ningún token durante
un periodo largo (en mi configuración, 1 000 pasos de entrenamiento),
se considera **muerta**.

En mi entrenamiento el colapso fue brusco: en el paso **1 190 de
12 207**, **21 780 de las 24 576 latentes (89 %)** estaban muertas.

La solución propuesta es una pérdida auxiliar llamada **AuxK**, que se
suma a la pérdida principal:

1. El SAE reconstruye el flujo residual con sus 32 latentes activas. Lo
   que le falta a la reconstrucción para igualar al original es el
   **residuo**.
2. Entre las **latentes muertas**, se toman las 384 con mayor activación
   en ese token.
3. Se les pide que reconstruyan ese residuo, y su error se añade a la
   pérdida con un coeficiente de $1/32$.

Así, las latentes muertas vuelven a recibir gradiente y aprenden a
representar precisamente **lo que las vivas no explican**. Cuando una se
vuelve útil, empieza a ganar un puesto entre las 32 activas. Con AuxK,
al final del entrenamiento solo quedaban **42 latentes muertas**.

Que AuxK estuviera activo mientras las latentes revivían es una
**correlación**, no una prueba de causa. Para establecer la causa repetí
el entrenamiento completo con los mismos datos, la misma semilla
aleatoria y los mismos hiperparámetros, **eliminando solo AuxK**.

<figure>
  <video src="assets/videos/AuxKAblationScene.mp4" controls muted playsinline preload="metadata"></video>
  <figcaption>Número de latentes muertas en los dos entrenamientos, paso a paso.</figcaption>
</figure>

Sin AuxK, el colapso ocurre igual (22 768 latentes muertas en el paso
1 310) y no se recupera: termina con 22 346. En el conjunto reservado:

| | Con AuxK | Sin AuxK |
|---|---:|---:|
| Latentes que se activan ≥ 1 vez en 520 000 tokens | **96,7 %** | 9,0 % |
| CE loss score | **98,4 %** | 96,3 % |

Sin AuxK, el SAE funciona con unas **2 200 latentes**. El resto de los
parámetros entrenados no contribuye a nada.

## Las latentes son interpretables, las neuronas no

Que un SAE reconstruya bien no implica que sus latentes correspondan a
conceptos reconocibles. Para medirlo hice una **evaluación ciega**, en
la que el evaluador no sabe qué está evaluando.

- Seleccioné al azar **120 latentes vivas** del SAE y **120 neuronas**
  del MLP del bloque 7, el último que escribe en el flujo residual antes
  de nuestro punto de lectura.
- Para cada una, extraje del conjunto reservado los **12 fragmentos de
  texto con mayor activación**.
- Los presenté mezclados, sin identificar, a **dos modelos de lenguaje
  distintos**, que puntuaron cada unidad de 1 (*sin patrón reconocible*)
  a 5 (*un único concepto, sin excepciones*), con la escala empleada en
  *Towards Monosemanticity* [3].

<figure>
  <video src="assets/videos/NeuronVsLatentScene.mp4" controls muted playsinline preload="metadata"></video>
  <figcaption>Una neurona y una latente, y la distribución de las 240 puntuaciones.</figcaption>
</figure>

Los dos evaluadores coincidieron casi por completo (correlación de
rangos de Spearman **0,95**). La probabilidad de observar una diferencia
así por azar es del orden de **10⁻²⁰** (prueba de Mann-Whitney).

Algunas latentes que obtuvieron un **5 con ambos evaluadores**, tal como
salieron de la muestra aleatoria:

- **#10274** — robar (`steal`, `stole`, `stolen`, `theft`).
- **#19814** — la palabra `dangerous`.
- **#17961** — nombres de mes dentro de una fecha (`On 17 March`).
- **#14748** — `Trek` y `Wars` inmediatamente después de `Star`.
- **#11659** — `Reuters` como firma de agencia.
- **#7840** — `played` en el sentido de *interpretar un papel*.
- **#5183** — el `of` de `the role of`.
- **#3433** — la negación tras un "you" genérico (`You can't control…`).

El diccionario contiene características **semánticas** (robo, peligro),
**sintácticas muy específicas** (el `of` que sigue a `role`) y de
**formato** (la firma de una agencia). Ninguna se especificó de
antemano.

## Las latentes tienen efecto causal

Que la latente **#10274** se active con textos sobre robos admite dos
lecturas:

1. Es **correlacional**: la dirección refleja la presencia del concepto,
   pero la red no la usa para decidir qué escribir.
2. Es **causal**: la red usa esa dirección para producir su salida.

Para distinguirlas hay que **intervenir**. La técnica se llama
**steering** (*dirección de la generación*). Mientras GPT-2 genera, se
suma al flujo residual la dirección de la latente en el diccionario,
escalada a su activación máxima observada, en todas las posiciones salvo
la primera.

El experimento necesita un **control**, porque sumar cualquier vector al
flujo residual altera la red y el texto podría cambiar solo por esa
perturbación. El control consiste en sumar una **dirección aleatoria**
de la misma magnitud. Si la latente no tiene un papel específico, ambas
intervenciones deberían producir el mismo efecto.

<figure>
  <video src="assets/videos/SteeringScene.mp4" controls muted playsinline preload="metadata"></video>
  <figcaption>El mismo comienzo y la misma semilla de muestreo, con y sin la latente #10274.</figcaption>
</figure>

Con el comienzo `The main thing to know is` y la misma semilla
aleatoria:

- **Sin intervención**: *…that after filing for bankruptcy, you will have to pay a penalty of up to $50,000…*
- **Con la latente #10274**: *…that he stole nearly $50,000 from the taxpayers of the state…*

Para cuantificarlo mejor, generé **80 continuaciones** de 40 tokens por
condición (5 comienzos distintos × 16 muestras) y conté cuántas
contenían alguna palabra del concepto, con una lista de palabras
**fijada antes** de ver los resultados:

| Latente | Sin intervención | Con la latente | Dirección aleatoria |
|---|---:|---:|---:|
| **#10274** robar   | 0 % | **32 %** | 3 % |
| **#19814** peligro | 0 % | **32 %** | 0 % |
| **#19443** color   | 4 % | **55 %** | 0 % |

Aun así, hay que dejar constancia de dos **limitaciones** claras:

- La intensidad tiene un óptimo. Con el doble de la activación máxima o
  más, el texto pierde coherencia y el concepto desaparece con él.
- No todas las latentes interpretables sirven para dirigir. Probé seis:
  la de "beber" no superó al control y la de "bromas" llegó al 15 %. La
  de meses aumentó su propia activación en el texto generado sin que
  aparecieran más nombres de mes que en el control.

## La longitud de contexto de entrenamiento importa

Entrené mi SAE con textos de **1 024 tokens**, la ventana de contexto
completa de GPT-2. Otros SAEs, como el de OpenAI o el de
[@JBloomAus](https://github.com/jbloomAus) (creador de SAELens), se
entrenaron con **64** y **128** tokens respectivamente, porque se
diseñaron para otros usos.

Evaluados con textos de 128 tokens, los tres rinden igual; con textos
de 1 024 tokens la historia es otra:

| SAE | Contexto de entrenamiento | CE loss score a 1 024 tokens |
|---|---:|---:|
| **Este trabajo** | 1 024 | **98,3 %** |
| OpenAI | 64 | 90,3 % |
| Joseph Bloom | 128 | 88,2 % |

Y además, el SAE de @JBloomAus pasa de activar 67 latentes por token
a **588**.

<figure>
  <video src="assets/videos/ContextLengthScene.mp4" controls muted playsinline preload="metadata"></video>
  <figcaption>Los tres SAEs evaluados con 128 y con 1 024 tokens.</figcaption>
</figure>

La explicación más probable es que la distribución de las activaciones
del flujo residual cambia con la posición en el texto, y un SAE que no
ha visto posiciones avanzadas no las representa bien. En la práctica,
**un SAE debe entrenarse con la longitud de contexto con la que se va a
usar**.

## Un error de medición que se evitó

La **varianza explicada** mide qué fracción de la variabilidad del flujo
residual captura la reconstrucción. Durante el entrenamiento salía
**0,94**; en la evaluación reservada, **0,85**.

La diferencia venía del **primer token de cada texto**. GPT-2 lo usa
como **sumidero de atención** (*attention sink*), una posición a la que
las capas de atención dirigen la atención que no necesitan en ningún
otro sitio. Su flujo residual tiene una norma unas **30 veces mayor**
que la del resto. Al incluirlo, domina el cálculo y hace que cualquier
SAE parezca mejor de lo que es. **Todas las cifras de este artículo lo
excluyen.**

## Límites de este trabajo

- Las puntuaciones de interpretabilidad son de **modelos de lenguaje**,
  no de personas. La concordancia entre dos evaluadores distintos es
  alta, pero pueden compartir sesgos.
- Solo se evaluaron las **activaciones máximas**. Una latente puede ser
  nítida en sus activaciones más altas y difusa en las intermedias.
- **Un modelo, un punto de lectura**. Los resultados son para GPT-2
  small tras el bloque 8. Templeton et al. (Anthropic, 2024) [4]
  mostraron que el método escala a Claude 3 Sonnet, pero no he
  reproducido eso.
- El **15 % de la varianza queda sin explicar**. Su efecto sobre las
  predicciones es pequeño (CE loss score del 98 %), pero es estructura
  que el diccionario no captura.

## Material abierto

- **Código** (entrenamiento, análisis y animaciones):
  [github.com/alexcerezo/minibeca](https://github.com/alexcerezo/minibeca)
- **SAE entrenado**:
  [huggingface.co/alexcerezo/sae-gpt2-small-l8-topk32](https://huggingface.co/alexcerezo/sae-gpt2-small-l8-topk32)
- **Curvas de entrenamiento**, con y sin AuxK:
  [W&B](https://wandb.ai/alexcerezocontreras-university-of-malaga/minibeca-xmihura)

```python
from huggingface_hub import snapshot_download
from sae_lens import SAE

sae = SAE.load_from_disk(snapshot_download("alexcerezo/sae-gpt2-small-l8-topk32"))
z = sae.encode(x)  # activaciones de las 24 576 latentes para cada token
```

## Bibliografía

1. N. Elhage et al., *Toy models of superposition*. Transformer Circuits Thread, 2022. <https://transformer-circuits.pub/2022/toy_model/index.html>
2. L. Gao et al., *Scaling and evaluating sparse autoencoders*. OpenAI, 2024. <https://cdn.openai.com/papers/sparse-autoencoders.pdf>
3. T. Bricken et al., *Towards monosemanticity: Decomposing language models with dictionary learning*. Transformer Circuits Thread, 2023. <https://transformer-circuits.pub/2023/monosemantic-features/index.html>
4. A. Templeton et al., *Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet*. Transformer Circuits Thread, 2024. <https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html>
5. J. Bloom, *SAELens*. GitHub, 2024. <https://github.com/jbloomAus/SAELens>
