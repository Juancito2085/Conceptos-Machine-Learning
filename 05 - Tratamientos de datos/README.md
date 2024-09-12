# Tratamiento datos

### Estandarización

La estandarización transforma los datos para que tengan una media de 0 y una desviación estándar de 1. Esto es útil cuando los datos siguen una distribución normal (gaussiana) o cuando se utilizan algoritmos que asumen esta distribución, como la regresión lineal y los métodos basados en distancia.

**Fórmula de Estandarización** (Z-Score):

$X_{est​}=\frac{X-\mu}{\sigma}​$

Donde:

- $X$: Valor original.
- $\mu$: Media de la característica.
- $\sigma$: Desviación estándar de la característica.

**Ventajas:**

Menos sensible a los valores atípicos en comparación con la normalización.
Útil para algoritmos que asumen una distribución normal de los datos.

**Desventajas:**

No mantiene la relación proporcional entre los valores originales.

**¿Cuándo se utiliza?**

Cuando los datos siguen una distribución normal o cuando utilizas algoritmos que asumen esta distribución, como la regresión lineal y los métodos basados en distancia.

### Normalización

La normalización ajusta los valores de las características para que estén en un rango específico, generalmente entre 0 y 1. Esto es útil cuando las características tienen diferentes escalas y quieres que todas tengan la misma importancia en el modelo.

**Fórmula de Normalización** (Min-Max Scaling):

$X_{norm}​=\frac{X_{max​}−X_{min}}{​X−X_{min​​}}$

Donde

- $X$: Valor original.
- $X_{min}$: Valor mínimo de la característica.
- $X_{max}$: Valor máximo de la característica.

**Ventajas:**

Mantiene la relación proporcional entre los valores originales.
Útil para algoritmos que no asumen una distribución específica de los datos, como K-Nearest Neighbors y redes neuronales.

**Desventajas:**

Sensible a los valores atípicos (outliers), ya que estos pueden distorsionar el rango.

**¿Cuándo se utiliza?**
Cuando los datos no siguen una distribución normal y quieres mantener la relación proporcional entre los valores. Ideal para algoritmos como KNN y redes neuronales.

### Codificación de variables categóricas

**One-Hot**: este método convierte cada categoría en una columna binaria (0 ó 1).

**Codificacion ordinal**: asigna un valor numérico a cada categoría según un orden específico.

**Codificación de frecuencia**: reemplaza cada categoría con la frecuencia de su aparición en el conjunto de datos.

**Codificación binaria**: convierte las categorías en representaciones binarias.

**Codificación de objetivo**: Reemplaza cada categoría con la media del objetivo para esa categoría.