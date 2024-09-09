# Aprendizaje supervisado

Vamos a recordar de manera muy sencilla que los modelos de aprendizaje supervisado se basan en el uso de un conjunto de datos, tanto de entrada como de salida, para entrenar estos modelos y poder clasificar datos o predecir resultados con precisión.

Los podemos separar en algoritmos de :
- Regresión: aquí los datos de entrada se utilizan para predecir valores númericos. Por ejemplo predecir el precio de una casa basándose en las características de las mismas.
- Clasificación: donde los datos de entrada se asignan a categorías las cuales el algoritmos a partir de nuevos datos de entrada tratará de clasificar la categoría a la que corresponden. Por ejemplo predecir si un correo electrónico es spam o no.


### Algoritmos de Regresión

A continuación veremos los algoritmos de regresión explicada de manera muy simple para luego profundizar en la sección correspondiente.

#### Regresión lineal
En la regresión lineal se busca modelar la realción entre la variable respuesta $y$ y una o más variables independientes $x$. 

Matemáticamente el modelo tendría la siguiente forma:

$Y\approx\beta_0+\beta_1X$

Donde 

$\beta_0$ es el intercepto (si lo pensamos en 2 dimensiones sería en el punto donde la recta corta el eje $y$)

$\beta_1$ es la pendiente de la recta (o dicho de manera más coloquial la pendiente de la recta)

Estos dos coeficientes explicados anteriormente son los parámetros del modelo los cuales se calculan a partir del entrenamiento con los datos.

El método más común de regresión simple es el método de mínimos cuadrados.

A continuación veremos una animación de como el modelo se va ajustando a los datos que tiene.

![regresion_lineal_animacion](/gif/ani_regresion_lineal.gif)

Y finalmente una imagen con todos los datos sería.

![regresion_lineal](/img/regresion_lineal.png )

#### Regresión Ridge

Este tipo de regresión es muy útil para analizar multiples variables que están altamente correlacionadas entre sí. Este método es especialmente útil para mitigar el problema de la multicolinealidad en la regresión lineal.

Otra de las ventajas que posee es mejorar la varianza de los coeficientes por lo que puede mejorar la capacidad de generalización del modelo. A continuación vemos una animación de este modelo.
![regresion_ridge_animacion](/gif/ani_regresion_ridge.gif)


En la siguiente figura vemos la regresión Ridge realizada a partir de los datos, básicamente es un regresión lineal con características diferentes a la regresión lineal simple.

![regrion_ridge](/img/regresion_ridge.png)
#### Regresión Lasso

La regresión Lasso es una técnica de regresión lineal que se utiliza para mejorar la precisión y la interpretabilidad de los modelos. 

Lasso puede reducir a cero los coeficientes de algunas variables, lo que efectivamente elimina esas variables del modelo. Esto ayuda a simplificar el modelo y a enfocarse en las variables más importantes.

Al penalizar los coeficientes, Lasso ayuda a reducir el sobreajuste, lo que mejora la capacidad de generalización del modelo.

Al eliminar algunas variables, el modelo resultante es más fácil de interpretar, ya que se enfoca solo en las variables más relevantes.

También este método puede manejar problemas de multicolinealidad al seleccionar solo una de las variables correlacionadas y reducir a cero las demás.

#### Elasticnet

Este modelo combina las penalizaciones de Lasso (L1) y Ridge (L2). Esto permite aprovechar las ventajas de ambos métodos, promoviendo la sparsidad y manejando la multicolinealidad.

Al igual que Lasso, Elastic Net puede reducir a cero los coeficientes de algunas variables, eliminándolas efectivamente del modelo. Esto ayuda a simplificar el modelo y a enfocarse en las variables más importantes.

Este método es particularmente útil cuando las variables independientes están altamente correlacionadas. A diferencia de Lasso, que puede seleccionar solo una de las variables correlacionadas, Elastic Net puede seleccionar un grupo de variables correlacionadas.

Al combinar las penalizaciones L1 y L2, Elastic Net puede encontrar un mejor equilibrio entre sesgo y varianza, mejorando la capacidad de generalización del modelo.

Elastic Net introduce dos parámetros de regularización: uno para la penalización L1 y otro para la penalización L2. Esto permite un control más fino sobre el grado de regularización aplicado.

#### Regresión lineal con Descenso de Gradiente Estocástico

El objetivo de la regresión lineal es encontrar la línea que mejor se ajuste a los datos, minimizando la suma de los errores cuadrados entre las predicciones y los valores reales.

Se utiliza la función de pérdida de error cuadrático medio (MSE), que mide la diferencia entre las predicciones del modelo y los valores reales.

Es un proceso iterativo que ajusta los parámetros en cada iteración, estos se actualizan en la dirección opuesta al gradiente de la función de pérdida con respecto a esos parámetros.
#### Regresión polinómica

Este regresión sirve para modelar relaciones no lineales entre variables independientes y la variable dependiente. En este caso se ajusta un polinomio de grado $n$ a los datos.

Matemáticamente generalizando el modelo se puede presentar de la siguiente manera:

$Y\approx\beta_0+\beta_1x^1+...+ \beta_nx^n$

Donde $\beta_0,\beta_1, ... ,\beta_n$ son los coeficientes del modelo

### Algoritmos de clasificación

Recordemos de manera muy rápida que los algoritmos de clasificación tiene como salida etiquetas o clases.
A continuación vamos a ver los mas usados en python

#### Regresión logística

Es un modelo estadístico utilizado para predecir la probabilidad de que ocurra un evento binario (es decir, un evento con dos posibles resultados, como “sí” o “no”, “verdadero” o “falso”) en función de una o más variables independientes.

La regresión logística estima la probabilidad de que ocurra un evento, con resultados que siempre están entre 0 y 1.

Utiliza la función logística (o sigmoide) para transformar la salida de una combinación lineal de las variables independientes en una probabilidad.

Aunque es un modelo lineal en los coeficientes, la transformación logística permite modelar relaciones no lineales entre las variables independientes y la probabilidad del evento.

La fórmula básica de la regresión logística es:

$logit(p)=\ln(\frac{p}{1-p})=\beta_0+\beta_1x_1+...+\beta_nx_n$

Donde

- $p$ es la probabilidad de que suceda el evento
- $\beta_0$ es el intercepto
- $\beta_1x_1+...\beta_nx_n$ son los coeficientes de las variables independientes.

#### Máquinas de soporte vectorial(SVM)
El objetivo principal de una SVM es encontrar el hiperplano que mejor separe las diferentes clases en el espacio de características. Este hiperplano es una línea (en 2D), un plano (en 3D) o un hiperplano (en dimensiones superiores).

Las SVM buscan maximizar el margen, es decir, la distancia entre el hiperplano y los puntos de datos más cercanos de cada clase. Estos puntos cercanos se llaman vectores de soporte.

Para problemas linealmente separables, las SVM encuentran un hiperplano lineal. Para problemas no linealmente separables, utilizan kernels para transformar los datos a un espacio de mayor dimensión donde un hiperplano lineal pueda separar las clases.

La ecuación del hiperplano de una SVM es:

$w.x+b=0$

Donde:

- $w$ es el vector de pesos

- $x$ es el vector de características

- $b$ es el sesgo

#### K-vecinos más cercanos(KNN)

Este algoritmo clasifica un punto de datos basado en la proximidad de sus vecinos más cercanos. La idea principal es que los puntos de datos similares estarán cerca unos de otros.

Funciona de la siguiente manera:

1.  Selección de K: Se elige un número ( K ) de vecinos más cercanos.

2. Cálculo de Distancias: Se calcula la distancia entre el punto de datos a clasificar y todos los puntos en el conjunto de entrenamiento. La distancia euclidiana es la más comúnmente utilizada.

3. Identificación de Vecinos: Se seleccionan los ( K ) puntos más cercanos.

4. Clasificación: Para problemas de clasificación, se asigna la clase más común entre los ( K ) vecinos. Para problemas de regresión, se toma el promedio de los valores de los ( K ) vecinos.

5. Aplicaciones: KNN se utiliza en diversas áreas como la detección de escritura manuscrita, reconocimiento de imágenes, y sistemas de recomendación, entre otros.

#### Árboles de decisión
Son utilizados tanto para tareas de clasificación como de regresión. 

Un árbol de decisión tiene una estructura jerárquica similar a un árbol, que consta de un nodo raíz, ramas, nodos internos y nodos hoja. El nodo raíz es el punto de partida y las ramas representan las decisiones o divisiones basadas en las características de los datos.

- Nodo raíz: El punto de inicio del árbol.
- Nodos internos: Representan decisiones basadas en una característica específica.
- Nodos hoja: Representan los resultados finales o las clases en las que se clasifican los datos.

El árbol de decisión divide los datos en subconjuntos más pequeños y homogéneos mediante decisiones basadas en las características de los datos. Este proceso se repite de manera recursiva hasta que se alcanza un nodo hoja.
#### Random Forest
Este método se basa en la creación de múltiples árboles de decisión y combina sus resultados para mejorar la precisión y evitar el sobreajuste.

El funcionamiento se puede resumir de la siguiente manera:

- Construcción de Árboles de Decisión: Se generan varios árboles de decisión a partir de diferentes subconjuntos de datos. Cada árbol se entrena de manera independiente.
- Votación o Promedio: Para problemas de clasificación, cada árbol “vota” por una clase y la clase con más votos es la predicción final. Para problemas de regresión, se toma el promedio de las predicciones de todos los árboles.
- Aleatoriedad: Se introduce aleatoriedad tanto en la selección de los datos como en la selección de las características (features) para cada árbol, lo que ayuda a crear árboles no correlacionados y mejora la robustez del modelo.

Este método es muy popular debido a su capacidad para manejar grandes conjuntos de datos y su flexibilidad para trabajar con diferentes tipos de datos y problemas.

#### Gradient Boosting

Es un método combina múltiples árboles de decisión de manera independiente, el Gradient Boosting construye el modelo de manera secuencial, donde cada nuevo árbol intenta corregir los errores de los árboles anteriores.

En este caso el funcionamiento es el siguiente:

- Inicialización: Se comienza con un modelo simple, como la media de los valores de salida en el caso de la regresión.
- Cálculo de Residuos: En cada iteración, se calculan los residuos (errores) del modelo actual.
- Ajuste de un Nuevo Árbol: Se ajusta un nuevo árbol de decisión a estos residuos.
- Actualización del Modelo: El nuevo árbol se agrega al modelo existente, ajustando su predicción para reducir los errores.
- Repetición: Este proceso se repite varias veces, cada vez ajustando un nuevo árbol a los residuos del modelo actualizado23.

Gradient Boosting es conocido por su capacidad para manejar diferentes tipos de funciones de pérdida, lo que lo hace muy adaptable a una variedad de problemas. Además, técnicas como el “shrinkage” (reducción de la tasa de aprendizaje) y el boosting estocástico se utilizan para mejorar la precisión y prevenir el sobreajuste.

#### Naive bayes

Es un algoritmo de machine learning basado en el teorema de Bayes, utilizado principalmente para tareas de clasificación. Es conocido por su simplicidad y eficiencia, especialmente en problemas de clasificación de texto, como el filtrado de spam y el análisis de sentimientos.

Los principios básicos son:


1 -  Teorema de Bayes: El algoritmo se basa en el teorema de Bayes, que describe la probabilidad de un evento, basado en el conocimiento previo de condiciones relacionadas con el evento. La fórmula es:
$P(A \mid B) = \frac{P(B \mid A)\cdot P(A)}{P(B) }$

donde:

$P(A \mid B)$ es la probabilidad de $A$ dado $B$.

$P(B \mid A)$ es la probabilidad de $B$ dado $A$.

$P(A)$ y $P(B)$ son las probabilidades de $A$ y $B$ respectivamente.



2 -Suposición de Independencia: Naive Bayes asume que las características (features) son independientes entre sí, lo cual rara vez es cierto en la práctica, pero aún así el algoritmo funciona sorprendentemente bien en muchos casos.

#### Redes neuronales artificiales

Las redes neuronales artificiales (ANN, por sus siglas en inglés) son modelos computacionales inspirados en el funcionamiento del cerebro humano. Estas redes están diseñadas para reconocer patrones y aprender de los datos.

**Componentes Principales de las RNA**

- Neuronas Artificiales: Las unidades básicas de una red neuronal, que reciben entradas, las procesan y generan una salida.
- Capas: Las neuronas se organizan en capas:
- Capa de Entrada: Recibe los datos iniciales.
- Capas Ocultas: Procesan las entradas a través de múltiples transformaciones.
- Capa de Salida: Genera la predicción final del modelo.
Pesos y Sesgos: Cada conexión entre neuronas tiene un peso que se ajusta durante el entrenamiento. Los sesgos ayudan a ajustar la salida de las neuronas.

**Funcionamiento**
- Propagación Hacia Adelante: Los datos de entrada se pasan a través de las capas de la red, multiplicándose por los pesos y aplicando funciones de activación para generar una salida.
- Función de Activación: Determina si una neurona debe activarse o no. Ejemplos comunes incluyen la función sigmoide, ReLU (Rectified Linear Unit), y tanh.
- Entrenamiento: Utiliza algoritmos como la retropropagación para ajustar los pesos y minimizar el error entre las predicciones de la red y los valores reales.

**Aplicaciones**

- Reconocimiento de Imágenes: Clasificación de objetos en imágenes.
- Procesamiento de Lenguaje Natural: Traducción automática, análisis de sentimientos.
- Predicción de Series Temporales: Pronóstico de ventas, análisis financiero.
- Las redes neuronales son especialmente poderosas en tareas donde los patrones son complejos y difíciles de definir con reglas explícitas

#### Clasificación de Vecinos Más Cercanos (Nearest Neighbors)

El método de los k vecinos más cercanos (KNN) es un algoritmo de aprendizaje supervisado utilizado tanto para clasificación como para regresión. Es conocido por su simplicidad y efectividad en una variedad de problemas.

**Principios Básicos**

- Proximidad: El algoritmo clasifica un nuevo punto de datos basado en la mayoría de votos de sus ( k ) vecinos más cercanos. Para problemas de regresión, se toma el promedio de los valores de los ( k ) vecinos más cercanos.
- Distancia: La distancia entre puntos de datos se calcula generalmente usando la distancia euclidiana, aunque se pueden usar otras métricas como la distancia de Manhattan.

**Funcionamiento**
- Almacenamiento de Datos: Durante la fase de entrenamiento, el algoritmo simplemente almacena todos los datos de entrenamiento.
- Clasificación: Para clasificar un nuevo punto, se calcula la distancia entre este punto y todos los puntos de entrenamiento. Luego, se seleccionan los ( k ) puntos más cercanos y se asigna la clase más común entre ellos (para clasificación) o se calcula el promedio (para regresión).


#### Análisis Discriminante Lineal y Cuadrático (LDA y QDA)

El Análisis Discriminante Lineal (LDA) y el Análisis Discriminante Cuadrático (QDA) son métodos de clasificación supervisada que buscan asignar observaciones a categorías predefinidas basándose en sus características.

**Análisis Discriminante Lineal (LDA)**

**Suposiciones**

LDA asume que las variables predictoras siguen una distribución normal y que las matrices de covarianza son iguales para todas las clases.

**Función Discriminante**

Utiliza una combinación lineal de las variables predictoras para separar las clases.

**Aplicaciones**

Es útil cuando las clases están bien separadas y se requiere un modelo estable, especialmente cuando el número de observaciones es bajo.

**Análisis Discriminante Cuadrático (QDA)**

**Suposiciones**
QDA también asume que las variables predictoras siguen una distribución normal, pero permite que las matrices de covarianza sean diferentes para cada clase.

**Función Discriminante**

Utiliza una combinación cuadrática de las variables predictoras, lo que permite una mayor flexibilidad en la separación de las clases.

**Aplicaciones**

Es más adecuado cuando las clases no están bien separadas y las matrices de covarianza son diferentes.

**Diferencias Clave**

- Complejidad del Modelo: LDA es más simple y menos flexible que QDA, ya que utiliza una combinación lineal de las variables. QDA, al permitir combinaciones cuadráticas, puede capturar relaciones más complejas.
- Estabilidad: LDA tiende a ser más estable cuando las clases están bien separadas y las matrices de covarianza son iguales. QDA es más flexible pero puede ser menos estable si las suposiciones no se cumplen.

Ambos métodos son herramientas poderosas en el análisis de datos y la clasificación, y la elección entre ellos depende de las características específicas de los datos y las suposiciones que se puedan hacer sobre ellos.