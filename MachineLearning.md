# 1. MACHINE LEARNING
Machine learning consiste en otorgar la habilidad de poder tomar decisiones a partir de datos sin estar programados para ello. 
Por ejemplo:
1. Saber si un correo es Spam o no. (tiene labels o etiquetas -> Sí o No).
2. Agrupar entradas de una web en función de su contenido (no tiene labels).

El primero de los casos corresponde a aprendizaje supervisado, y el segundo no.

El último consiste en descubrir patrones ocultos a partir de datos sin etiquetar.

# 1.1 Worokflow
1. Extraer las características (features).
2. Separar el dataset a dataset de entrenamiento y dataset de test.
3. El dataset de entramiento debe ser entrenado: entrenar el modelo.
    - Se introducen los datos en un modelo de ML (para ser entrenados).
    - Se produce un modelo de entrenamiento a partir de este paso. Podemos suponer que el modelo es usable, pero antes de poder usarlo éste debe ser previamente evaluado.
4. El modelo obtenido debe ser evaluado. Para ello se usan los datos que hemos extraído del paso 2 (el dataset de Test). Para entrenar el modelo, debemos utilizar datos que el modelo generado no haya visto

<br>
<br>

# 2. SUPERVISED LEARNING
El objetivo es encontrar la variable que llamamos objetivo a partir de una serie de variable con valores dadas. Básicamente, consiste una tomar una observación y asignar una etiqueta (label). 

La columna/variable que buscamos se llama target (objetivo) y las variables que se nos dan son conocidas como features o  predictor (características).

Existen dos tipos de aprendizaje supervisado en función de la variable objetivo que buscamos:

1. Si el target es una variable discreta, es decir, que consiste en categorías nos referimos a éste como classification. Asignar una categoría.
2. Si nos encontramos con un target cuya variable es continua, es decir, que puede ser cambiante (como el precio de una casa), nos encontramos con un caso de regression.

El objetivo principal de aprendizaje supervisado es automatizar aplicaciones que consumen mucho tiempo o que emplean demasiados recursos, así como realizar predicciones sobre el futuro.

La librería más conocida para realizar esta tarea es scikit-learn/sklearn.

## 2.1 SL Clasificado
Truco para saber si nos encontramos ante este tipo o regresión, anotar cuáles son los posibles targets y a partir de ahí, ver si es un valor discreto o continuo.

    from sklearn import datasets
    
    iris = datasets.load_iris(), que es un Bunch (como un dict)
    
    iris.keys(), que nos da data, target_names, DESCR, feature_names, target.

    iris.data
    irist.target
    iris.data.shape

Las muestras están en filas y las características en columnas: iris.data.shape nos devolverá (x, y) dodnde x serán las muestras e y las características.

Para hacer un EDA hay que pasar a Dataset.

    x = iris.data
    y = iris.value

    df = pd.DataFrame(X, columns=iris.feature_names)

**EDA** es el proceso de organizar, trazar y resumir un conjunto de datos.

Ejemplo guay de EDA usando seaborn (sns) para realizar un plot.

    plt.figure()
    sns.countplot(x='education', hue='party', data=df, palette='RdBu')
    plt.xticks([0,1], ['No', 'Yes'])
    plt.show()


## 2.2 Realizar el clasificado
Nuestro objetivo es sacar datos etiquetados de los que tenemos, que no lo están, a partir de datos que ya han sido previamente etiquetados.

A estos datos ya etiquetados se les 
llama **datos entrenados**.

A partir de aquí se construye el clasificador.

### 2.3 Tipo de clasificador KNN
    K- Nearest Neighbors: Consiste en observar los datos más cercanos al que se está evaluando. En función de la cantidad de valores que haya de un tipo u otro, se asignará el valor de la mayoría al que se está estudiando.

    Se tomará una dimensión en función del valor de K: si k=3, tomamos los 3 puntos más cercanos.


Todos los modelos de sklearn buscan:
1. Implemantar un serie de algoritmos para aprender y predecir.
2. Guardar información a partir de los datos aprendidos.

Entrenar un modelo es proporcional a ajustarlo a los datos. 

Se utiliza el método .fit() para realizar esta tarea. También contamos con el método .predict() para averiguar el método que estamos utilizando.

Las características (features) deben ser continuas y no pueden haber lagunas de datos.

Hay funciones interesantes para separar training y test data. 
    train_test_split

## 2.2 SL Regresión
En este caso los valores qus buscamos son continuos. Debemos operar con arrays de tipo numpy (hay métodos para pasar). 

En caso de no contar con el formato idóneo, pasar por reshape().

    reg = LinearRegression(), siendo esta función de sklearn

Una regresión siempre funciona con el siguiente esquema:

    y = ax + b

donde **y** es el target y **x** es una única feature

a, b son los parámetros del modelo

La historia está en saber cómo elegimos a y b. Esto se realiza definiendo funciones de error (coste) e una línea concreta y elegir aquella línea que minimice el error. 

Se utiliza OLS, una función que minimiza la suma de los cuadrados de los residuos (distancia del punto a la recta). 

fit() realiza esta tarea por debajo.

El método para evaluar si nuestro modelo es bueno o es malo se denomina R squared.

La metodología es muy similar a la clasificación, cambiando la variable reg.

La Api de sklearn funciona de la misma manera que las mátemáticas cuando tenemos que trazar una recta de estas características, siendo las **a** las features y **b** el target.

    y = a1·x1 + a2·x2 + a3·x3 +... + an·xn + b

### 2.2.1 Cross Validation
Utilizar el método de R-squared no es demasiado efectivo a la hora de generalizar, puesto que el resultado al evaluear el modelo es dependiente de cómo se separan los datasets (train y test).

Para evitar esto, echamos mano de cross validation, que realiza la operación varias veces (las que le indiquemos en cv) y nos otorga tantos resultados como cifra hayamos fijado. De ahí, sacamos la media y obtenemos un resultado mucho más preciso.

#### 2.2.2 Regularización
Muchas veces podemos caer en el error de contar con un modelo sobrenetrenado (overfitting) o infraentrenado. Esto se debe a que no se las características suficientes para producir una predicción eficaz. Esto significa un fallo de nuestro modelo al generalizar -encajar- el conocimiento que pretendemos que adquieran.

- Overfitting: Aportar datos demasiado particulares a un modelo, de tal forma que no es capaz de generalizar cuando llega un dato mínimamente diferente. Por ejemplo, entrenamos al modelo con muestras de 10 perros de color marrón. Despúes, aportamos un perro de color blanco. Ante la pregunta de si estamos ante un perro, la respuesta es negativa puesto que esta muestra se sale del patrón de las que tenemos.

- Underfitting: Le damos una única muestra (un perro marrón) y le preguntamos si una nueva muestra (un perro blanco) cumple con los requisitos. Lógicamente, la respuesta será no y el motivo es que no contamos con las muestras suficientes para entrenar a nuestro modelo.
