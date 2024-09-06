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

#### Bosques aleatorios

#### Gradient Boosting

#### Naive bayes

#### Redes neuronales artificiales

#### Clasificación de Vecinos Más Cercanos (Nearest Neighbors)

#### Análisis Discriminante Lineal y Cuadrático (LDA y QDA)
