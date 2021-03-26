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

    .score calcula la precisión

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

        reg = LinearRegression()


        cv_scores = cross_val_score(reg, característica, target, cv=5)


#### 2.2.2 Regularización
Muchas veces podemos caer en el error de contar con un modelo sobrenetrenado (overfitting) o infraentrenado. Esto se debe a que no se las características suficientes para producir una predicción eficaz. Esto significa un fallo de nuestro modelo al generalizar -encajar- el conocimiento que pretendemos que adquieran.

- Overfitting: Aportar datos demasiado particulares a un modelo, de tal forma que no es capaz de generalizar cuando llega un dato mínimamente diferente. Por ejemplo, entrenamos al modelo con muestras de 10 perros de color marrón. Despúes, aportamos un perro de color blanco. Ante la pregunta de si estamos ante un perro, la respuesta es negativa puesto que esta muestra se sale del patrón de las que tenemos.

- Underfitting: Le damos una única muestra (un perro marrón) y le preguntamos si una nueva muestra (un perro blanco) cumple con los requisitos. Lógicamente, la respuesta será no y el motivo es que no contamos con las muestras suficientes para entrenar a nuestro modelo.

Existen dos metodologías de regularización importantes:
- Ridge: utilizamos una función OLS (Ordinary Least Squares) para para ajustar los datos (fit) a una función multiplicada por un coeficiente alfa. 

        alfa · sumatorio de la caracterísitca al cuadrado.

    siendo alfa el parámetro que controla la complejidad del sistema.
    
    Un alfa demasiado alto puede conllevar a overfitting y uno muy bajo a underfitting.

        ridge = Ridge(alfa = 0.1, normalize=True)


- Lasso: Similar, solo que el producto es alfa · el sumatorio de los varlores absolutos de la característica.

        lasso = Lasso(alpha=0.1, normalize=True)

        lasso.fit(X,y)

        lasso_coef = lasso.fit(X,y).coef_

        con X siendo la característica elegida e y el target

<br>

## 2.3 Cuánto de bueno es nuestro modelo

Dentro de la Clasificación, (2.1), podemos utilizar otras métricas además de a precisión (accuracy, el porcentaje de muestras correctamente detectadas).
Ejemplo de los emails de spam. Se necesitan otras metodologías más efectivas. Para ello, necesitamos comprender las variantes involucradas.

- Verdaderos Positivos (TP): el modelo lo clasifica como positivo y es positivo.
- Verdaderos Negativos (TN): el modelo lo clasifica como negativo y es negativo.
- Falsos Positivos (FP): el modelo lo clasifica como positivo y es negativo.
- Falsos Negativos (FN): el modelo lo clasifica como negativo y es positivo.

Sabiendo esto, procedemos a comprender cuáles son las diferentes métricas que podemos utilizar para llevar a cabo 
1. Precisión (“Accuracy”): el porcentaje de aciertos totales
        
        A = TP + TN / TP + FP + TN + FN

2. Exactitud ( “Precision”): el porcentaje de predicciones positivas que son acertadas

        P= TP / TP + FP

3. Exhaustividad (“Recall”): el porcentaje de valores positivos que identificados por en la predicción

        R= TP / TP + FN
<br>

### 2.3.1 Regresión Logística

Se utiliza en el ámbito de clasificación y se emplea para mostrar probabilidades. Solo asigna los valores 0 y 1.
- Si el valor p es superior a 0'5, se asgina el valor 1.
- En caso contrario, se asigna un 0.

El umbral se suele fijar en 0,5 porque está en el centro, pero éste puede variar.

- Si se pone a 0, todos slos valores serán 1 (el ratio de TP es el mismo que el de FP, es decir, 1).
- Si se pone a 1, todos los valores serán 0 (el ratio de TP es el mismo que el de FP, es decir, 0).

La variación de los puntos que probamos para conseguir todos los umbrales posibles, se llama curva **ROC** (Reciever Operating Characteristic curve).

Una **curva ROC** (curva de característica operativa del recepto) es un gráfico que muestra el rendimiento de un modelo de clasificación en todos los umbrales de clasificación. Esta curva representa dos parámetros:

- Tasa de verdaderos positivos (Exhaustividad)
- Tasa de falsos positivos (FP / FP + TP)

Los informes de clasificación y las matrices de confusión son excelentes métodos para evaluar cuantitativamente el rendimiento de los modelos, mientras que las curvas ROC proporcionan una forma de evaluar visualmente los modelos.

Una curva ROC representa TPR frente a FPR.

### 2.3.2 AUC

Dada un curva ROC, debemos buscar la métrica que más nos interese.

Para evaluar un modelo de regresión logística es ineficiente utilizar umbrales, por lo que se echa mano del algoritmo AUC (Area Under ROC Curve).

### 2.3.3 Hyperparameter tuning

Este tipo de parámetros se utilizan para evitar el over y underfittin (como alfa en Ridge y Lasso, o n en KNN vecinos).

Se trata de un parámetro que no puede ser elegido antes de predecir el modelo y debe tener un valor correcto. Para ello, se deben probar varios valores, ajustarlos de forma separa y observar el rendimiento para quedarnos con el que mejor funcione. 
- Utilizar la validación cruzada es esencial (cross validation).



### 2.3.4 Evaluación final

Es importante utilizar los datos de entrenamiento para poder llevar a cabo la cross validation. Si utilizo todos los datos de los que dispongo, no sería efectivo, puesto que estamos evaluando un modelo con datos que previamente ha visto.

<br>
<br>

## 2.3 Árboles de decisión

Se trata de secuencias de perguntas if else sobre características de carácter individual. 

El objetivo es inferir las etiquetas de las clases.

La diferencua principal con los modelos lineales, es que los árboles de decisión pueden capturar relaciones no liniales entre entidades características (features) y etiquetas (labels). No necesita estar en la misma escala que el resto de las variables.

Cada pregunta if-else involucra una característica del dataset.

### 2.3.1 Arquitectura
Se trata de una estructura jerárquica con nodos, siendo éstos donde se realiza la pregunta if-else o la predeicción. 

Tres tipos de nodos:
- Raíz (Root)
- Nodo interno
- Hoja (Leaf)


El árbol toma sus decisiones en función de la IG (Information Gain), que depende de forma intrínseca de la característica y de la "split point". Éstos crecen de manera recursiva, por lo que dependen del estado de sus predecesores.

En cada nodo, el árbol se hace un pregunta involucrando una característica y un split point, que elegirá en función de la IG. 

Existe una fórmula que mide la "impuridad" de un nodo: I(nodo), que puede calcularse echando mano de varios métodos.

Si IG es 0, estamos ante una hoja (donde se lleva a cabo la decisión). Si nos encontramos en el nivel 2 de un árbol con profundidad 2, estamos de nuevo ante una hoja aunque no sea 0.

### 2.3.2 Árbol de Decisión para regresión

