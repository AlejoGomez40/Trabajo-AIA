# ===============================================
# EJERCICIO 2: IMPLEMENTACIÓN ÁRBOLES DE DECISIÓN
# ===============================================


# En este ejercicio pedimos implementar en python un algoritmo de aprendizaje para árboles 
# de decisión. Los árboles de decisión que trataremos serán árboles binarios, en los que
# en cada nodo interior se pregunta por el valor de un atributo o característica dada, 
# y si ese valor es mayor o menor que un valor umbral dado. Este es el mismo tipo de árbol 
# de decisión que se  manejan en Scikit Learn. 

# Se puede obtener información de este tipo de árboles en la entrada "Decision Trees"
# del manual de Scikit Learn. También en la práctica del Titanic hecha en clase.

# Se propone la implementación de un clasificador basado en árboles de
# de decisión, entrenado usando el algoritmo CART, similar al que implementa 
# la clase DecisonTree de Scikit Learn, pero con ALGUNAS VARIANTES, que indicaremos más
# adelante.

# Los árboles de decisión están formados por nodos. Usar la siguiente clase para la
# implementación de los nodos:

# --------------------
# Funciones Auxiliares
# --------------------

# Calcula las proporciones de cada clasificacion
def prop_y(y):
    _, conteos = np.unique(y, return_counts=True)
    probabilidades = conteos / y.size
    return probabilidades

# Calcula la aleatoridad de datos en un nodo
def entropia(y):
    if y.size == 0:
        return 0.0
    probabilidades = prop_y(y) # Usamos la funcion anterior para obtener las probabilidades
    return -np.sum(probabilidades * np.log2(probabilidades)) # Fórmula de entropía sobre el array

# Calcula la ganancia de informacion al dividir los datos
def ganancia_informacion(y_padre, y_izq, y_der):
    peso_izq = y_izq.size / y_padre.size
    peso_der = y_der.size / y_padre.size
    entropia_hijos = (peso_izq * entropia(y_izq)) + (peso_der * entropia(y_der))
    return entropia(y_padre) - entropia_hijos

# -----------------------
# Declaración de la clase
# -----------------------

class Nodo:
    def __init__(self, atributo=None, umbral=None, izq=None, der=None,distr=None,*,clase=None):
        self.atributo = atributo # Indice de la columna por la que pregunta el nodo
        self.umbral = umbral # Valor numerico de corte
        self.izq = izq # Referencia al Nodo hijo izquierdo
        self.der = der # Referencia al Nodo hijo derecho
        self.distr= distr # Diccionario con el conteo de clases en ese punto
        self.clase = clase # Si es hoja, almacena la prediccion. Si es interior, es None
        
    def es_hoja(self):
        return self.clase is not None

# Pasamos a describir los distintos atributos de esta clase:

# - atributo: el atributo por el que se pregunta en el nodo. Referenciaremos a cada
#   atributo POR EL ÍNDICE DE SU POSICIÓN (el número de columna).
# - umbral: es el valor umbral por el que se pregunta en el nodo. Si la instancia tiene un
#   valor de atributo menor o igual que el umbral, se sigue por el subárbol izquierdo. En
#   caso contrario, por el subárbol derecho.
# - izq: es el nodo raiz del subárbol izquierdo.
# - der: el nodo raiz del subárbol derecho.
# - distr: es un diccionario cuyas claves son las posibles clases, y cuyos valores son
#   cuántos ejemplos del conjunto de entrenamiento correspondientes al nodo hay de cada
#   clase. Cuando decimos "ejemplos correspondientes al nodo" queremos decir aquellos que
#   cumplen todas las condiciones (desde la raiz) que llevan a ese nodo.
# - clase: Si el nodo es una hoja, es la clase que predice. Si no es una hoja, este valor es None.



# Lo que sigue es una descripción del algoritmo que se pide implementar para la
# construcción de un árbol de decisión. En principio describiremos la versión básica y más
# conocida, y posteriormente indicaremos las peculiaridades y variantes que pedimos
# introducir a esta versión básica.

# Supondremos que recibimos un conjunto de entrenamiento X,y y además dos valores max_prof
# y min_ejemplos_nodo_interior, que nos van a servir como condiciones adicionales para
# dejar de expandir un nodo. El algoritmo se define recursivamente y tiene además un
# argumento adicional prof (inicialmente 0), con la profundidad del nodo actual.  

# CONSTRUYE_ARBOL(X,y,min_ejemplos_nodo_interior,max_prof,prof=0):

# 1. SI prof es mayor o igual que max_prof, 
#       o el número de ejemplos de X es menor que min_ejemplos_nodo_interior,
#       o en X todos los ejemplos son de la misma clase:
#       ENTONCES:
#          Devolver un nodo hoja con la distribución de clases en X,
#                    y con la clase mayoritaria en X
# 2. EN OTRO CASO:
#        encontrar el MEJOR atributo A y el mejor umbral u para ese atributo
#        y particionar en dos tanto X como y:
#            * X_izq, y_izq los ejemplos cuyo valor de A es menor o igual que u
#            * X_der, y_der los ejemplos cuyo valor de A es mayor que u
#        Llamadas recursivas:
#            A_izq=CONSTRUYE_ARBOL(X_izq,y_izq,min_ejemplos_nodo_interior,max_prof,prof+1)
#            A_der=CONSTRUYE_ARBOL(X_der,y_der,min_ejemplos_nodo_interior,max_prof,prof+1)
#        Devolver un nodo interior con el atributo y umbral seleccionado,
#                 con la distribución de clases de X, y con A_izq y A_der
#                 como hijos izquierdo y derecho respectivamente.


# Lo anterior es la descripción básica. A continuación indicamos una serie de variantes y
# cuestiones adicionales que se le piden a esta implementación concreta:

# - Consideraremos la posibilidad de restringir los atributos a usar en el árbol a un
#   número de atributos dado n_atrs. Ese subconjunto de atributos se seleccionará
#   aleatoriamente al principio de la construcción del aŕbol y será el mismo para todos
#   los nodos.
#   Por ejemplo, si el dataset tiene 15 atributos y le damos n_atrs=9, al comienzo de la
#   construcción del árbol seleccionamos aleatoriamente 9 atributos, y ya en los nodos del
#   árbol solo podrán aparecer alguno de esos 9 atributos. Nótese que si n_atrs es igual
#   al total de atributos, tendríamos la versión estándar del algoritmo.
#   NOTA: téngase en cuenta que a diferencia de lo que ocurre en la versión clásica de
#   Random Forests, no sorteamos los atributos en cada nodo, sino que hay un único sorteo
#   inicial para todo el árbol.

# - A la hora de elegir el mejor atributo y umbral para la partición de los nodos
#   interiores, usar el criterio de mejor GANANCIA DE INFORMACIÓN (en particular, NO USAR GINI).

# - La principal carga computacional de este algoritmo se debe a la cantidad de candidatos a
#   mejor atributo y mejor umbral que hay que evaluar en cada nodo, para decidir cuál es
#   la mejor partición. El hecho de limitar el número de atributos candidatos (como se ha
#   descrito más arriba), va en esa dirección. 
#   Otra manera es limitar también los posibles valores umbrales a considerar
#   para cada atributo. Para ello, en la implementación que se pide actuaremos en dos
#   sentidos:
#      (a) Considerar solo como candidatos a umbral los puntos medios entre cada par de 
#         valores consecutivos del atributo en los que hay cambio de clase, para los
#         ejemplos correspondientes a ese nodo.
#         Por ejemplo, si ordenados los valores del atributo A en orden creciente, hay un
#         ejemplo con valor v1 de A y clase C1 y a continuación otro ejemplo con valor v2
#         en A y clase C2 distinta de C1, entonces (v1+v2)/2 es un posible valor umbral
#         candidato. El resto de valores NO se considera candidato.

#      (b) En cada nodo, para elegir los umbrales candidatos correspondientes a un atibuto,
#         no considerar todos los ejemplos que corresponden a ese nodo, sino 
#         sólo  una proporción de los mismos, seleccionada aleatoriamente. La proporción a
#         considerar se da en un parámetro prop_umbral.
#         Por ejemplo, si prop_umbral es 0.7 y el conjunto de ejemplos correspondientes al
#         nodo es de 200 ejemplos, entonces aplicaremos el proceso de selección de
#         umbrales candidatos descrito en (a) considerando sólo un suconjunto de 140
#         ejemplos seleccionado aleatoriamente de entre esos 200.  



# Con las descripciones anteriores, ya podemos precisar lo que se pide en eset apartado. 
# Se pide implementar una clase ArbolDecision con el siguiente formato:
  

class ArbolDecision:
    def __init__(self, min_ejemplos_nodo_interior=5, max_prof=10,n_atrs=10,prop_umbral=1.0):
        self.min_ejemplos_nodo_interior = min_ejemplos_nodo_interior
        self.max_prof = max_prof
        self.n_atrs = n_atrs
        self.prop_umbral = prop_umbral
        
        # Estas dos variables se inicializan vacías porque dependen de los datos de entrada
        self.raiz = None # Almacenará el Nodo inicial tras el entrenamiento
        self.atributos_seleccionados = None # Almacenará los índices de las columnas permitidas
               
    def entrena(self, X, y):
        # 1. Determinamos el número total de características (columnas) en la matriz X
        total_atributos = X.shape[1] 
        
        # 2. Sorteamos un subconjunto aleatorio de índices de columnas sin repetición
        if self.n_atrs < total_atributos:
            self.atributos_seleccionados = random.sample(range(total_atributos), self.n_atrs)
        else:
            self.atributos_seleccionados = list(range(total_atributos))

        # 3. Empieza recursividad desde el nivel de profundidad 0 y asigna el resultado a la raíz
        self.raiz = self.construye_arbol(X, y, prof=0)

    def construye_arbol(self, X, y, prof):
        # 1. Distribucion de los datos entrantes
        valores, conteos = np.unique(y, return_counts=True)
        distr = dict(zip(valores, conteos))

        # 2. Evaluacion de las reglas de parada
        limite_profundidad = prof >= self.max_prof
        pocos_ejemplos = X.shape[0] < self.min_ejemplos_nodo_interior
        nodo_puro = len(valores) == 1

        # 3. Ejecución de la parada (Creamos la hoja)
        if limite_profundidad or pocos_ejemplos or nodo_puro:
            indice_clase_mayoritaria = np.argmax(conteos)
            clase_ganadora = valores[indice_clase_mayoritaria]
            return Nodo(distr=distr, clase=clase_ganadora)
        
        # 4. Encontramos la mejor columan y el numero exacto por donde cortar
        mejor_atributo, mejor_umbral = self.encuentra_mejor_division(X,y)

        if mejor_atributo is None:
            indice_clase_mayoritaria = np.argmax(conteos)
            clase_ganadora = valores[indice_clase_mayoritaria]
            return Nodo(distr=distr, clase=clase_ganadora)

        # 5. Creamos las máscaras y dividimos físicamente AMBAS matrices (X e y)
        mascara_izq = X[:, mejor_atributo] <= mejor_umbral
        mascara_der = X[:, mejor_atributo] > mejor_umbral

        X_izq, y_izq = X[mascara_izq], y[mascara_izq]
        X_der, y_der = X[mascara_der], y[mascara_der]

        # 6. Llamadas recursivas (usando el método propio de la clase y sumando nivel)
        hijo_izq = self.construye_arbol(X_izq, y_izq, prof + 1)
        hijo_der = self.construye_arbol(X_der, y_der, prof + 1)

        # 7. Empaquetamos todo en un Nodo de tipo interior (sin parámetro 'clase')
        return Nodo(
            atributo=mejor_atributo, 
            umbral=mejor_umbral, 
            izq=hijo_izq, 
            der=hijo_der, 
            distr=distr
        )


    def encuentra_mejor_division(self, X, y):
        mejor_ganancia = -1
        mejor_atributo = None
        mejor_umbral = None

        # Recorremos solo las columnas que sorteamos al principio en entrena
        for atributo in self.atributos_seleccionados:
            columna_datos = X[:, atributo]
            
            # No miramos todas las filas, sacamos una muestra
            n_filas = X.shape[0]
            n_muestras = int(n_filas * self.prop_umbral)

            # Seleccionamos las filas aleatoriamente
            indices_muestra = random.sample(range(n_filas), n_muestras)
            valores_muestra = columna_datos[indices_muestra]
            clases_muestra = y[indices_muestra]

            # Ordenar
            indices_ordenados = np.argsort(valores_muestra)
            valores_ordenados = valores_muestra[indices_ordenados]
            clases_ordenadas = clases_muestra[indices_ordenados]

            for i in range(1, len(valores_ordenados)):
                if clases_ordenadas[i] != clases_ordenadas[i-1]:
                    umbral_candidato = (valores_ordenados[i] + valores_ordenados[i-1])/2
                    mascara_izq = columna_datos <= umbral_candidato
                    mascara_der = columna_datos > umbral_candidato
                    y_izq = y[mascara_izq]
                    y_der = y[mascara_der]
                    ganancia = ganancia_informacion(y, y_izq, y_der)
                    
                    if ganancia > mejor_ganancia:
                        mejor_ganancia = ganancia
                        mejor_atributo = atributo
                        mejor_umbral = umbral_candidato
            
        return mejor_atributo, mejor_umbral

    def clasifica(self, X):
        if self.raiz is None:
            raise ClasificadorNoEntrenado("El modelo debe ser entrenado antes de usarse.")
        # 1. Creamos una lista vacía para guardar la respuesta de cada fila
        predicciones = []

        # 2. Recorremos la matriz X fila por fila
        for i in range(X.shape[0]):
            # Extraemos los datos del ejemplo actual
            ejemplo = X[i, :]
            
            # Empezamos el recorrido desde la cima del árbol
            nodo_actual = self.raiz

            # 3. Navegamos hacia abajo mientras el nodo actual NO sea una hoja
            while not nodo_actual.es_hoja():
                # Miramos el valor que tiene este ejemplo en la columna que pide el nodo
                valor_atributo = ejemplo[nodo_actual.atributo]

                # Comparamos con el umbral para decidir el camino
                if valor_atributo <= nodo_actual.umbral:
                    nodo_actual = nodo_actual.izq
                else:
                    nodo_actual = nodo_actual.der

            # 4. Al salir del bucle while, hemos llegado a una hoja. 
            # Guardamos la clase ganadora de esa hoja en nuestra lista.
            predicciones.append(nodo_actual.clase)

        # 5. Convertimos la lista final en un array de NumPy y lo devolvemos
        return np.array(predicciones)

    def clasifica_prob(self, x):
        if self.raiz is None:
            raise ClasificadorNoEntrenado("El modelo debe ser entrenado antes de usarse.")
        ejemplo = x
        # Empezamos el recorrido desde la cima del árbol
        nodo_actual = self.raiz
        # 3. Navegamos hacia abajo mientras el nodo actual NO sea una hoja
        while not nodo_actual.es_hoja():
           # Miramos el valor que tiene este ejemplo en la columna que pide el nodo
            valor_atributo = ejemplo[nodo_actual.atributo]
            # Comparamos con el umbral para decidir el camino
            if valor_atributo <= nodo_actual.umbral:
                nodo_actual = nodo_actual.izq
            else:
                nodo_actual = nodo_actual.der

        # 5. Convertimos la lista final en un array de NumPy y lo devolvemos
        total_ejemplos_hoja = sum(nodo_actual.distr.values())
        probabilidades = {clase: conteo / total_ejemplos_hoja for clase, conteo in nodo_actual.distr.items()}
        
        return probabilidades

    def imprime_arbol(self, nombre_atrs, nombre_clase):
        if self.raiz is None:
            raise ClasificadorNoEntrenado("El modelo debe ser entrenado antes de usarse.")
        # Función auxiliar interna para manejar la recursividad
        def _recorre_nodo(nodo, nivel):
            # 5 espacios por cada nivel de profundidad
            espacio = "     " * nivel

            if nodo.es_hoja():
                # Formato de hoja: "Sobrevive: 1 -- {1: 10}"
                print(f"{espacio}{nombre_clase}: {nodo.clase} -- {nodo.distr}")
            else:
                # Nombre de la columna
                nombre = nombre_atrs[nodo.atributo]
                
                # Rama izquierda (menor o igual, con 3 decimales)
                print(f"{espacio}{nombre} <= {nodo.umbral:.3f}")
                _recorre_nodo(nodo.izq, nivel + 1)
                
                # Rama derecha (mayor, con 3 decimales)
                print(f"{espacio}{nombre} > {nodo.umbral:.3f}")
                _recorre_nodo(nodo.der, nivel + 1)

 #       # Validación de seguridad
#      if self.raiz is None:
 #           print("El árbol no ha sido entrenado.")
  #          return

        # Iniciamos en la raíz con nivel 0
        _recorre_nodo(self.raiz, 0)



#  El constructor tiene los siguientes argumentos de entrada:

#     + min_ejemplos_nodo_interior: mínimo número de ejemplos del conjunto de 
#       entrenamiento en un nodo del árbol que se aprende, para que se considere 
#       su división.  
#     + max_prof: profundidad máxima del árbol que se aprende.
#     + n_atrs: número de atributos candidatos a considerar en cada partición
#     + prop_umbral: proporción de ejemplos a considerar cuando se buscan los 
#       umbrales candidatos.    
  
#      

# * El método entrena tiene como argumentos de entrada:
#   
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento 
#        y su clasificación esperada, respectivamente.
#     

# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos. 

# * Método clasifica_prob: recibe UN EJEMPLO y devuelve un diccionario con la predicción
#   de probabilidad de pertenecer a cada clase. Esa probabilidad se calcula como la
#   proporción de ejemplos de clase en la distribución del nodo hoja que da la
#   predicción.

# * Método imprime_arbol: recibe la lista de nombres de cada atributo (columnas) y el
#   nombre del atributo de clasificación, e imprime el árbol de decisión aprendido 
#   (ver ejemplos más abajo) [SUGERENCIA: hacerlo con una función auxiliar recursiva] 


# Si se llama al método de clasificación, o al de impresión, antes de entrenar el modelo,
# se debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass

        


