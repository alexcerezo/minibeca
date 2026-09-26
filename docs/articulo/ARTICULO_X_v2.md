# Abrí el cerebro de GPT-2 y encontré su diccionario

### Qué hay dentro de una IA, por qué no se entiende a simple vista y cómo conseguí leerlo. Explicado desde cero.

---

Cuando le preguntas a ChatGPT por qué ha dicho algo, te da una explicación que suena muy razonable. El problema es que no hay forma de saber si es verdad. Es como preguntarle a alguien por qué ha tomado una decisión: te contará una historia, pero eso no significa que sea lo que realmente pasó en su cabeza.

Con las personas no podemos hacer otra cosa. Con las IA, sí: su "cerebro" es un montón de números que podemos mirar uno a uno. Si algún día vamos a dejar que estos sistemas tomen decisiones importantes, queremos poder abrirlos y comprobar qué están pensando de verdad, en lugar de fiarnos de lo que nos cuentan.

Este artículo cuenta cómo lo intenté con GPT-2, un modelo pequeño y público de OpenAI, qué encontré y cómo me aseguré de no estar engañándome. No hace falta saber nada de inteligencia artificial para seguirlo.

**Lo que encontré, en cuatro frases:**

1. Construí un "diccionario" que traduce los números internos de GPT-2 a conceptos, y funciona tan bien como el que publicó OpenAI usando menos recursos.
2. Las piezas de ese diccionario se entienden: el 87 % representan una sola idea clara. Las neuronas originales del modelo, solo el 34 %.
3. Esas piezas no solo *describen* lo que piensa el modelo: si las activas a mano, cambias lo que escribe. Activando la pieza de "color", más de la mitad de los textos pasan a hablar de colores.
4. Hay un truco de entrenamiento sin el cual el 91 % del diccionario se queda inservible. Lo comprobé repitiendo todo el experimento sin él.

---

## Parte 1. Lo mínimo que necesitas saber

### Cómo "piensa" un modelo de lenguaje

Un modelo como GPT-2 hace una sola cosa: leer un texto y adivinar cuál es la siguiente palabra. Lo repite una y otra vez, y así escribe.

Para hacerlo, no trabaja con letras sino con números. Cada trozo de palabra (lo que se llama un *token*: "gato", "ando", una coma...) se convierte en una lista de **768 números**. Esa lista va pasando por **12 etapas** llamadas *capas*, y en cada una el modelo la modifica un poco, añadiendo información: qué palabra es, qué función tiene en la frase, de qué va el texto, qué tono tiene...

Al final de las 12 capas, esos 768 números contienen todo lo que el modelo "sabe" sobre ese momento del texto, y los usa para elegir la siguiente palabra.

Así que, si queremos saber qué está pensando el modelo, tenemos que mirar esos 768 números en alguna de las capas. Yo miré los de la **capa 8**, más o menos a dos tercios del recorrido.

### El problema: los números no significan nada por separado

Lo natural sería pensar que cada uno de esos números (o cada "neurona" del modelo) se encarga de algo concreto: una para los animales, otra para el pasado, otra para el enfado...

No es así. Esto es lo que pasa cuando cojo una neurona real de GPT-2 y miro los textos en los que más se enciende (marco entre « » la palabra exacta):

> "Up« **in** » heaven" · "about 1« **,** »000 officers" · "Eric H« **ov** »de"

Una preposición, la coma de los miles y una sílaba de un apellido. No hay ningún concepto común. Y no es una neurona rara: en mi evaluación, **dos de cada tres neuronas son así**, un cajón de sastre.

### Por qué pasa esto: la maleta demasiado pequeña

El lenguaje tiene muchísimos más conceptos que 768. Hay miles de temas, nombres propios, tiempos verbales, estilos, formatos... ¿Cómo caben en 768 números?

Imagina que tienes que meter la ropa de un mes en una maleta de fin de semana. No puedes darle a cada prenda su propio hueco, así que las doblas y las encajas unas sobre otras. Funciona porque nunca te pones toda la ropa a la vez: cada día solo sacas unas pocas prendas.

GPT-2 hace lo mismo. Guarda cada concepto no en una neurona, sino en una **combinación** de muchas neuronas, y deja que esas combinaciones se solapen un poco. Como en cada frase solo aparecen unos pocos conceptos a la vez, el solapamiento casi no molesta. Los investigadores de Anthropic (la empresa de Claude) llamaron a esto **superposición** en 2022.

> 🎬 **[Vídeo 1: SuperpositionScene]** *Cinco conceptos compartiendo un espacio donde en teoría solo caben dos. Mientras solo uno esté activo, los demás apenas le molestan.*

Para el modelo es una solución brillante. Para quien quiere entenderlo es un desastre: cada neurona es un trocito de muchos conceptos a la vez. Mirar una neurona es como intentar adivinar qué hay en la maleta tocando un solo punto de la tela.

---

## Parte 2. La herramienta: un diccionario

### La idea

Si el problema es que los conceptos están mezclados, la solución es **desmezclarlos**. La herramienta que usé se llama *Sparse Autoencoder* (SAE), pero la idea se entiende mejor con una analogía.

Piensa en un batido de frutas. Lo pruebas y sabe a "batido", pero no sabes qué lleva. Un SAE es como una máquina que, a partir del batido, te dice la receta: "30 % plátano, 20 % fresa, 10 % mango". Para eso necesita:

1. **Una lista de ingredientes posibles** (el diccionario). En mi caso, **24 576 "ingredientes"**. A cada uno lo llamaré *latente*.
2. **Una regla de oro**: en cada receta solo puede usar **32 ingredientes** de esos 24 576. Ni uno más.
3. **Una comprobación**: con esa receta de 32 ingredientes tiene que ser capaz de volver a preparar el batido original, y que sepa igual.

La regla de oro es la clave. Si la máquina pudiera usar todos los ingredientes a la vez, volvería a mezclarlo todo. Al obligarla a usar muy pocos, la única forma de acertar es que cada ingrediente represente algo concreto y reutilizable: un concepto.

Nadie le dice a la máquina qué conceptos buscar. Se le dan 100 millones de palabras de texto de internet, y ella sola va descubriendo qué ingredientes necesita para reconstruir lo que pasa dentro de GPT-2.

> 🎬 **[Vídeo 2: SAEAnatomyScene]** *Entran 768 números. De 24 576 latentes, solo 32 se encienden. A partir de esas 32 se reconstruyen los 768 números originales.*

### Cómo sé si funciona

Aquí está el truco para no engañarse. Hice esta prueba:

1. Dejo que GPT-2 lea un texto normal y mido lo bien que adivina la siguiente palabra.
2. Luego, en la capa 8, **le quito sus 768 números y le pongo en su lugar la reconstrucción del diccionario**, la hecha con solo 32 ingredientes. Y vuelvo a medir.

Si el diccionario ha capturado todo lo importante, GPT-2 ni se entera del cambio. Si se ha dejado cosas por el camino, GPT-2 empezará a equivocarse más.

Para tener una referencia, también probé el caso extremo: **borrar del todo** esos números. Con estos tres casos puedo calcular qué parte del daño evita el diccionario:

| Qué le doy a GPT-2 en la capa 8 | Error al adivinar la siguiente palabra |
|---|---:|
| Sus números originales | 3,59 |
| **La reconstrucción de mi diccionario** | **3,71** |
| Nada (números borrados) | 11,11 |

*(El error se mide en una unidad llamada "nats". No importa qué significa exactamente: menos es mejor.)*

Borrar la capa hace que el error se dispare de 3,59 a 11,11. Usar mi diccionario solo lo sube a 3,71. Es decir, **el diccionario conserva el 98,4 % de lo que hace falta**. A esa cifra la llamaré *puntuación de reconstrucción*.

Todas estas mediciones las hice con textos que el diccionario **nunca había visto** durante el entrenamiento, como un examen con preguntas nuevas. Y usé exactamente el mismo examen para comparar mi diccionario con dos diccionarios públicos de referencia: el de **OpenAI** y el de **Joseph Bloom**.

Bloom es un investigador dedicado a entender por dentro los modelos de lenguaje. Creó SAELens, la herramienta gratuita que más gente usa para construir y estudiar este tipo de diccionarios, y en 2024 publicó diccionarios para todas las capas de GPT-2 que se han convertido en un punto de comparación habitual.

| | Tamaño del diccionario | Ingredientes por receta | Puntuación de reconstrucción |
|---|---:|---:|---:|
| **El mío** | 24 576 | 32 | **98,4 %** |
| OpenAI | 32 768 | 32 | 98,4 % |
| Joseph Bloom | 24 576 | 67 de media | 98,0 % |

Empato con OpenAI usando un diccionario un 25 % más pequeño, y supero a Bloom usando la mitad de ingredientes por receta.

---

## Parte 3. Lo que encontré

### Hallazgo 1: sin un truco concreto, el diccionario se muere

Durante el entrenamiento pasa algo curioso. Cada vez que el diccionario elige sus 32 ingredientes, los que no salen elegidos no aprenden nada en ese paso. Si un ingrediente pierde muchas veces seguidas, se queda atrás, cada vez tiene menos opciones de ser elegido, y acaba **muerto**: no se vuelve a usar nunca.

Es lo que les pasa a los jugadores de un equipo que nunca salen al campo: no mejoran, así que siguen sin salir.

En mi entrenamiento, esto ocurrió de golpe. Tras unos 1 200 pasos de entrenamiento, **21 780 de los 24 576 ingredientes (el 89 %) estaban muertos**.

La solución que propuso OpenAI en 2024 se llama **AuxK**. La idea es darles a los ingredientes muertos una tarea propia: "intentad explicar lo que los 32 elegidos no han conseguido explicar". Así siguen practicando aunque no jueguen, y en cuanto son útiles, vuelven a entrar en el equipo.

Con AuxK, el diccionario fue resucitando poco a poco hasta quedarse con **solo 42 ingredientes muertos** al final del entrenamiento.

Pero ojo: que AuxK estuviera activo y los ingredientes revivieran no demuestra que AuxK fuera la causa. Podría haber pasado igual por otros motivos. Para comprobarlo, **repetí el entrenamiento entero con todo idéntico (mismos datos, mismo punto de partida, mismos ajustes) salvo una cosa: sin AuxK**.

> 🎬 **[Vídeo 3: AuxKAblationScene]** *Dos entrenamientos idénticos salvo por un detalle. Uno resucita, el otro no.*

Sin AuxK, el colapso ocurre igual (22 768 muertos) y **nunca se recupera**: termina con 22 346. Los resultados en textos nuevos:

| | Con AuxK | Sin AuxK |
|---|---:|---:|
| Ingredientes que se usan alguna vez (en 520 000 palabras) | **96,7 %** | 9,0 % |
| Puntuación de reconstrucción | **98,4 %** | 96,3 % |
| Error extra que sufre GPT-2 al usar el diccionario | **0,12** | 0,28 |

Sin AuxK, el diccionario sobrevive con unos 2 200 ingredientes y tira el resto a la basura. Y GPT-2 sufre más del doble de error al usarlo.

**Qué significa**: AuxK no es un detalle técnico opcional. Sin él, el 91 % del diccionario que entrenas no sirve para nada, y ni siquiera te das cuenta si solo miras la reconstrucción, que sigue pareciendo decente.

### Hallazgo 2: los ingredientes se entienden; las neuronas, no

Que el diccionario reconstruya bien no garantiza que sus ingredientes signifiquen algo para un humano. Podrían ser igual de caóticos que las neuronas. Así que lo comprobé con una **evaluación a ciegas**, como una cata de vinos sin etiqueta:

1. Elegí al azar **120 ingredientes de mi diccionario** y **120 neuronas de GPT-2** (de la capa 7, que es la que escribe justo donde está el diccionario).
2. Para cada uno saqué los 12 fragmentos de texto en los que más se enciende.
3. Lo mezclé todo y se lo di a **dos evaluadores automáticos distintos** (dos modelos de IA diferentes), sin decirles qué era cada cosa. Tenían que puntuar de 1 ("no veo ningún patrón") a 5 ("representa un único concepto nítido"), siguiendo la escala que usó Anthropic en su investigación.

> 🎬 **[Vídeo 4: NeuronVsLatentScene]** *A la izquierda, una neurona de GPT-2. A la derecha, un ingrediente del diccionario. Después, las 240 notas.*

| | Neuronas de GPT-2 | Ingredientes del diccionario |
|---|---:|---:|
| Nota media (de 1 a 5) | 3,1 | **4,3** |
| Con nota 4 o 5 (se entienden bien) | 34 % | **87 %** |
| Con nota 5 (un solo concepto, sin excepciones) | 7 % | **50 %** |

Los dos evaluadores coincidieron casi por completo, y la diferencia es tan grande que la probabilidad de que sea casualidad es prácticamente cero (del orden de una entre cien trillones).

Estos son algunos de los ingredientes que sacaron un 5 en ambos evaluadores. Salieron del sorteo al azar; no los he elegido yo:

- **Robar**: se enciende con *steal, stole, stolen, theft*.
- **Peligro**: la palabra *dangerous*.
- **Meses dentro de fechas**: el *March* de "On 17 March", el *May* de "By May 1".
- ***Trek*** y ***Wars*** justo después de ***Star***.
- **Agencia de noticias**: *Reuters* cuando aparece como firma de una noticia.
- **Actuar**: *played* en el sentido de interpretar un papel ("played Hamlet"), no de jugar.
- **Una construcción concreta**: el *of* de "the role **of**".
- **La negación tras un "tú" genérico**: "You **can't** control…".

Hay de todo: ideas (robo, peligro), piezas de gramática muy específicas (el *of* después de *role*) y convenciones de formato (la firma de Reuters). **Nadie le enseñó al diccionario ninguno de estos conceptos.** Los descubrió solo, porque eran la forma más eficiente de describir lo que pasa dentro de GPT-2.

### Hallazgo 3: si tocas un ingrediente, cambias lo que escribe el modelo

Que un ingrediente se encienda con textos sobre robos puede significar dos cosas:

- Que **detecta** robos, como un termómetro que marca la temperatura sin influir en ella.
- Que el modelo lo **usa** para decidir qué escribir, como el termostato que sí la cambia.

Para distinguirlo hay que intervenir. El experimento es este: mientras GPT-2 escribe, **subo a mano el volumen de un ingrediente** y miro qué pasa con el texto.

Y aquí viene el control importante: subir el volumen de *cualquier cosa* altera al modelo, así que el texto podría cambiar solo por el ruido. Por eso repito el experimento empujando en una **dirección al azar con exactamente la misma fuerza**. Si el ingrediente no tiene nada especial, el empujón aleatorio debería producir el mismo efecto.

> 🎬 **[Vídeo 5: SteeringScene]** *El mismo comienzo de frase y el mismo azar, con y sin el ingrediente de "robar".*

Un ejemplo real. Le doy a GPT-2 el comienzo *"The main thing to know is"* ("Lo principal que hay que saber es"):

- **Sin tocar nada**: *"…that after filing for bankruptcy, you will have to pay a penalty of up to $50,000…"* ("…que tras declararte en bancarrota tendrás que pagar una multa de hasta 50 000 dólares…")
- **Subiendo el ingrediente de "robar"**: *"…that he **stole** nearly $50,000 from the taxpayers of the state…"* ("…que **robó** casi 50 000 dólares a los contribuyentes del estado…")

La misma cifra, el mismo arranque, pero ahora la historia es un robo.

Otro ejemplo con el mismo comienzo, esta vez con el ingrediente de "color". Sin tocar nada, GPT-2 escribe sobre una liga de fútbol americano universitario: *"…that the Pac-12 West Region is a special one"*. Con el ingrediente subido: *"…that the Pac, as shown on the picture above, was also the primary **color scheme** for their print"* ("…era también la **combinación de colores** principal de su impresión").

Para medirlo en serio generé **80 textos por cada nivel de volumen** (5 comienzos de frase distintos × 16 textos cada uno) y conté cuántos mencionaban el concepto. Las palabras que contaban como "mención" las fijé antes de ver ningún resultado, para no hacer trampas sin querer.

| Ingrediente | Sin tocar nada | Subiéndolo (mejor volumen) | Empujón aleatorio de la misma fuerza |
|---|---:|---:|---:|
| Robar | 0 % | **32 %** | 3 % |
| Peligro | 0 % | **32 %** | 0 % |
| Color | 4 % | **55 %** | 0 % |

**Qué significa**: estos ingredientes no son solo etiquetas que les ponemos desde fuera. Son **palancas** que el modelo usa de verdad para decidir qué decir.

Dos matices que no quiero esconder:

- **Hay un punto óptimo.** El mejor resultado sale con un volumen parecido al máximo que ese ingrediente alcanza de forma natural. Si subes el doble o más, GPT-2 empieza a escribir sin sentido y el concepto desaparece junto con todo lo demás. Es como subir tanto el volumen de un altavoz que solo se oye distorsión.
- **No todos los ingredientes funcionan como palanca.** Probé seis. El de "beber" no movió nada por encima del empujón aleatorio; el de "bromas" solo llegó al 15 %; y el de meses cambió lo que pasaba dentro del modelo sin que aparecieran nombres de meses en el texto. Que un ingrediente se entienda no garantiza que sirva para dirigir al modelo.

### Hallazgo 4: un diccionario solo funciona con textos tan largos como los que ha visto

GPT-2 puede leer textos de hasta 1 024 tokens (unas 750 palabras). Mi diccionario lo entrené con textos de esa longitud. Los de OpenAI y Bloom se entrenaron con fragmentos mucho más cortos (64 y 128 tokens), porque se pensaron para otros usos.

Con textos cortos, los tres empatan. Con textos largos, no:

> 🎬 **[Vídeo 6: ContextLengthScene]** *Los mismos tres diccionarios, evaluados con textos cortos y con textos largos.*

| | Entrenado con textos de | Puntuación con textos cortos (128) | Puntuación con textos largos (1 024) |
|---|---:|---:|---:|
| **El mío** | 1 024 | 98,4 % | **98,3 %** |
| OpenAI | 64 | 98,4 % | 90,3 % |
| Joseph Bloom | 128 | 98,0 % | 88,2 % |

El de Bloom, que usaba 67 ingredientes por receta, pasa a usar **588** con textos largos, y sus reconstrucciones se alejan tanto del original que acierta menos que una respuesta fija que siempre devolviera el valor medio. El mío apenas se inmuta.

La explicación más probable: lo que pasa dentro de GPT-2 al principio de un texto es distinto de lo que pasa en la palabra número 800. Un diccionario que solo ha visto principios de texto no sabe describir el resto. No es un fallo de esos diccionarios, que se hicieron para otra cosa, pero sí una advertencia práctica: **si quieres analizar textos largos, entrena con textos largos.**

---

## Parte 4. Siendo honestos

### Lo que esto no demuestra

- **Las notas de interpretabilidad las pusieron IAs, no personas.** Dos evaluadores distintos coinciden casi del todo, pero podrían compartir los mismos sesgos.
- **Solo miré los momentos en que cada ingrediente se enciende con más fuerza.** Un ingrediente puede ser muy nítido en su pico y más difuso cuando se enciende poco. Anthropic lo estudia con más detalle; yo no lo he hecho aquí.
- **Es un modelo pequeño y una sola capa.** GPT-2 tiene 124 millones de parámetros; los modelos actuales tienen cientos de miles de millones. Anthropic demostró que esta técnica también funciona en Claude 3 Sonnet, pero mis cifras concretas son de este modelo y esta capa.
- **Queda un 15 % sin explicar.** El diccionario reconstruye el 85 % de la variación de esos 768 números. Lo que falta no afecta mucho a GPT-2 (de ahí el 98 %), pero es estructura que todavía no entendemos.

### Una trampa en la que casi caigo

Mientras entrenaba, las métricas automáticas decían que el diccionario explicaba el 94 % de la variación. Con mi examen independiente salió el **85 %**.

La diferencia venía de la primera palabra de cada texto. GPT-2 la usa como una especie de "cajón de sastre" interno y sus números son unas 30 veces más grandes que los del resto. Como es tan grande, domina las cuentas y hace que cualquier diccionario parezca mejor de lo que es. Todas las cifras de este artículo la excluyen. Lo cuento porque es exactamente el tipo de error que hace que un resultado parezca mejor de lo que es sin que nadie haya hecho trampa a propósito.

---

## Por qué importa

Si un modelo tiene una pieza interna para "peligro", podemos hacer tres cosas con ella: **ver** cuándo la activa, **subirla** y **apagarla**. Eso es exactamente lo que necesitamos para auditar sistemas de IA: no preguntarles qué piensan y fiarnos de la respuesta, sino mirarlo directamente.

Este trabajo no descubre nada que los grandes laboratorios no hubieran anticipado. Lo que aporta es una réplica completa, hecha con un presupuesto mínimo, que incluye comprobaciones que a menudo faltan: repetir el entrenamiento sin AuxK para demostrar que es la causa, comparar a ciegas contra las neuronas originales, controlar los experimentos con empujones aleatorios y medir los diccionarios públicos en el mismo examen. Y todo está abierto para que cualquiera pueda repetirlo o rebatirlo.

## Todo es abierto

- **Código** (entrenamiento, análisis y animaciones): [github.com/alexcerezo/minibeca](https://github.com/alexcerezo/minibeca)
- **El diccionario ya entrenado**: [huggingface.co/alexcerezo/sae-gpt2-small-l8-topk32](https://huggingface.co/alexcerezo/sae-gpt2-small-l8-topk32)
- **Curvas de entrenamiento** (con y sin AuxK): [proyecto de W&B](https://wandb.ai/alexcerezocontreras-university-of-malaga/minibeca-xmihura)

Si programas en Python, cargarlo son tres líneas:

```python
from huggingface_hub import snapshot_download
from sae_lens import SAE
sae = SAE.load_from_disk(snapshot_download("alexcerezo/sae-gpt2-small-l8-topk32"))
```

Y `sae.encode(x)` te dice, para cada palabra, cuáles de los 24 576 ingredientes están encendidos.

---

## Glosario rápido

- **Modelo de lenguaje**: programa que aprende a predecir la siguiente palabra de un texto. GPT-2, ChatGPT y Claude lo son.
- **Token**: trozo de texto con el que trabaja el modelo; una palabra corta o parte de una larga.
- **Capa**: cada una de las etapas por las que pasa la información dentro del modelo. GPT-2 small tiene 12.
- **Neurona**: cada uno de los números que calcula el modelo en una capa.
- **Superposición**: guardar muchos más conceptos que neuronas, repartiendo cada concepto entre muchas neuronas.
- **Sparse Autoencoder (SAE)**: el "diccionario". Traduce los números del modelo a una combinación de pocos conceptos.
- **Latente** (o "ingrediente"): cada entrada del diccionario.
- **Latente muerta**: una que nunca se usa.
- **AuxK**: truco de entrenamiento que da trabajo a las latentes muertas para que revivan.
- **Steering** (dirigir): subir o bajar una latente a mano para cambiar lo que escribe el modelo.

---

## Detalles técnicos (para quien los quiera)

- **Punto de lectura**: flujo residual de GPT-2 small antes del bloque 8 (`blocks.8.hook_resid_pre`), donde existen SAEs públicos de referencia.
- **Arquitectura**: SAE TopK (Gao et al., OpenAI 2024), `k = 32`, expansión 32× (24 576 latentes), pérdida AuxK con coeficiente 1/32.
- **Datos**: 100 M tokens de OpenWebText, secuencias de 1 024 tokens. Evaluación en el último fragmento del dataset, que el entrenamiento no toca.
- **Puntuación de reconstrucción**: *CE loss score* = (error borrando − error con SAE) / (error borrando − error original). Pérdida extra con textos de 128 tokens: 0,12 nats (mío), 0,12 (OpenAI), 0,15 (Bloom); con 1 024 tokens: 0,14, 0,81 y 0,99.
- **Evaluación ciega**: 120 latentes vivas y 120 neuronas MLP de la capa 7, 12 contextos máximos cada una, dos evaluadores LLM, escala de *Towards Monosemanticity*. Correlación de Spearman entre evaluadores 0,95; Mann-Whitney p ≈ 10⁻²⁰; diferencia de medias +1,25 (IC 95 % [1,03; 1,47]).
- **Steering**: se suma `c · max_act · dirección` al flujo residual en todas las posiciones salvo la primera, con `c` ∈ {0,5; 1; 1,5; 2; 3}. El mejor resultado sale con `c = 1`. Control: dirección aleatoria de la misma norma.

---

*Este trabajo es mi entregable para las **Minibecas de XMihura**. Gracias a XMihura por apostar por que la investigación rigurosa en seguridad de IA también puede hacerse fuera de los grandes laboratorios.*

*Referencias: Elhage et al., "Toy Models of Superposition" (Anthropic, 2022) · Bricken et al., "Towards Monosemanticity" (Anthropic, 2023) · Templeton et al., "Scaling Monosemanticity" (Anthropic, 2024) · Gao et al., "Scaling and evaluating sparse autoencoders" (OpenAI, 2024).*
