# Trabajo AIA — Árboles de Decisión, Random Forest y Transfer Learning con CNNs

Trabajo práctico de la asignatura **Ampliación de Inteligencia Artificial** (Grado en Ingeniería Informática, Universidad de Sevilla), realizado en pareja.

El proyecto está dividido en dos partes independientes.

## Parte I — Árboles de decisión y Random Forest desde cero

Implementación **en Python puro con NumPy, sin usar scikit-learn**, de un clasificador de árbol de decisión y un ensemble de Random Forest, entrenados y evaluados sobre varios datasets reales.

- **`particion_entr_prueba`**: partición aleatoria y **estratificada** de un dataset en entrenamiento/prueba, preservando las proporciones de clase.
- **Cálculo de entropía y ganancia de información** como criterio de división de nodos.
- **Clase `ArbolDecision`**: construcción recursiva del árbol, con selección aleatoria de un subconjunto de atributos en cada nodo (parámetro `n_atrs`), búsqueda del mejor atributo/umbral de corte por ganancia de información, y reglas de parada configurables (profundidad máxima, mínimo de ejemplos por nodo, pureza del nodo). Incluye métodos de predicción (`clasifica`), predicción probabilística (`clasifica_prob`) e impresión legible del árbol aprendido (`imprime_arbol`).
- **Clase `RandomForest`**: ensemble de N árboles entrenados con *bagging* (muestreo con reemplazo) y votación mayoritaria para la predicción final.
- **Evaluación y ajuste de hiperparámetros** sobre varios datasets: Titanic (supervivencia), votos del Congreso de EE.UU. (predicción de partido), cáncer de mama (Wisconsin), crédito bancario, Adult Income (predicción de renta), reconocimiento de dígitos manuscritos (MNIST en formato texto) y clasificación de sentimiento en críticas de IMDB.
- Preprocesado de datos categóricos con `OrdinalEncoder` y carga/vectorización de imágenes de dígitos desde archivos de texto.

## Parte II — Transfer Learning con CNNs (TensorFlow/Keras)

Clasificación de imágenes de flores (5 clases: rosas, margaritas, dientes de león, girasoles, tulipanes) mediante redes neuronales convolucionales, comparando tres enfoques de complejidad creciente:

1. **CNN básica** entrenada desde cero.
2. **CNN con técnicas de regularización** (data augmentation, dropout) para reducir el overfitting del modelo anterior.
3. **Transfer Learning** sobre una red preentrenada, ajustando las capas finales al dataset de flores para maximizar el accuracy de validación.

Cada modelo se evalúa y compara con el anterior en accuracy de validación, ilustrando el impacto progresivo de cada técnica.

## Stack

`Python`, `NumPy`, `pandas`, `scikit-learn` (solo para preprocesado: `OrdinalEncoder`), `TensorFlow` / `Keras`.

## Estructura del repositorio

```
trabajo_aia_25_26_parte_I.py     # Árboles de decisión y Random Forest (implementación propia)
trabajo_aia_25_26_parteII.ipynb  # Transfer Learning con CNNs
carga_datos.py                   # Carga de los datasets base (iris, titanic, votos, cáncer, IMDB)
datos/                           # Datasets: crédito, adult, dígitos (MNIST en texto)
```

## Nota

Este es un trabajo académico evaluado en dos partes independientes (implementación desde cero y transfer learning), realizado junto con Carlos Gamito Moreno como parte de la asignatura.
