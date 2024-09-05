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

![regresion_linea_animacion](/gif/ani_regresion_lineal.gif)

Y finalmente una image con todos los datos sería.

![regresion_lineal](/img/regresion_lineal.png "hola")

#### Regresión Ridge

#### Regresión Lasso

#### Elasticnet

#### Regresión lineal con Descenso de Gradiente Estocástico

#### Regresión polinómica

Este regresión sirve para modelar relaciones no lineales entre variables independientes y la variable dependiente. En este caso se ajusta un polinomio de grado $n$ a los datos.

Matemáticamente generalizando el modelo se puede presentar de la siguiente manera:

$Y\approx\beta_0+\beta_1x^1+...+ \beta_nx^n$

Donde $\beta_0,\beta_1, ... ,\beta_n$ son los coeficientes del modelo