# 1. CRÉDITO 
# print(X_credito.shape)
# buscar_y_evaluar("CRÉDITO", X_train_credito, y_train_credito,None,None,X_test_credito, y_test_credito,
#                  lista_n_arboles=[5, 10], #Probamos con 5 y 10 por ejemplo ya que no es un dataset muy grande X_credito.shape = (650,6) 
#                  lista_prop_muestras=[1.0], #Al no ser muy grande podemos permitirnos que sea del mismo tamaño que el original
#                  lista_min_ejemplos_nodo_interior=[5], #5 que es un punto intermedio entre 2, que sería demasiado específico y el ruido afectaria, y tampoco es muy grande ya que el dataset es pequeño
#                  lista_max_prof=[5, 10], #Probamos con 5, 10 que no son valores muy grandes para evitar sobreajuste
#                  lista_n_atrs=[3, 6], #Probamos con todos los atributos y con la mitad
#                  lista_prop_umbral=[1.0]) #Como hay pocos datos podemos evaluar el 100% de los cortes

# # 2. ADULT DATASET 
# print(X_adult.shape)
# buscar_y_evaluar("ADULT", X_train_adult, y_train_adult,None,None,X_test_adult, y_test_adult,
#                  lista_n_arboles=[5], #Solo probamos 5, ya que este dataset es mucho más grande y 10 tardará mucho X_adult.shape = (32561, 12)
#                  lista_prop_muestras=[0.5], # Usamos solo la mitad de los datos en cada árbol para aligerar la memoria y forzar la diversidad en el bosque
#                  lista_min_ejemplos_nodo_interior=[20], #Al tener tantas filas, subimos el mínimo a 20 para forzar paradas tempranas y no perder tiempo aislando ruido
#                  lista_max_prof=[5, 10], #Probamos con 5 y 10 para no colapsar la RAM por sobreajuste
#                  lista_n_atrs=[4, 8], #Hemos investigado y una buena heurística suele ser la raiz cuadrada del numero de características (en nuestro caso 4) y hemos probado tambien el doble de este resultado
#                  lista_prop_umbral=[0.1]) #Como evaluar los cortes en casi 30.000 filas es lento y costoso, miramos solo el 10% aleatorio.

# # 3. IMDB 
# print(X_train_imdb.shape)
# print(X_test_imdb.shape)
# buscar_y_evaluar("IMDB", X_train_imdb, y_train_imdb,None,None, X_test_imdb, y_test_imdb, #X_imdb.shape = 2400,632)
#                  lista_n_arboles=[15,20], #Probamos con 10, pero los resultados no eran buenos asi que aumentamos
#                  lista_prop_muestras=[0.8], 
#                  lista_min_ejemplos_nodo_interior=[5,10], 
#                  lista_max_prof=[15], # Probamos con 5 y 10, pero los resultados no eran del todo buenos, asi q aumentamos
#                  lista_n_atrs=[24,100,150], #Probamos con 24 y el doble, pero no eran buenos asi que probamos con mayores
#                  lista_prop_umbral=[0.1])

# # 4. DÍGITOS 
# print(X_train_dg.shape)
# buscar_y_evaluar("DÍGITOS", X_train_dg, y_train_dg, X_valid_dg, y_valid_dg, X_test_dg, y_test_dg,
#                  lista_n_arboles=[10], 
#                  lista_prop_muestras=[0.8], 
#                  lista_min_ejemplos_nodo_interior=[10], 
#                  lista_max_prof=[10], 
#                  lista_n_atrs=[28, 56, 150], 
#                  lista_prop_umbral=[0.1])