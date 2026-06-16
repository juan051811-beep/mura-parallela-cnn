# Grid Search para selección de hiperparámetros

Esta rama contiene la implementación de la búsqueda de hiperparámetros para tres modelos de clasificación utilizados en el proyecto de diagnóstico asistido mediante inteligencia artificial para radiografías óseas.

## Modelos evaluados

Se evaluaron los siguientes modelos:

- SVM (Support Vector Machine)
- Random Forest
- Regresión Logística

La búsqueda de hiperparámetros se realizó utilizando una muestra estratificada del 30% del conjunto de características extraídas mediante CNN, manteniendo la proporción entre clases normales y anormales.

## Paralelización

La búsqueda de hiperparámetros fue implementada utilizando la librería `multiprocessing` de Python.

Se comparó el tiempo de ejecución de una versión secuencial contra versiones paralelas utilizando:

- 2 procesos
- 4 procesos
- 6 procesos
- 8 procesos

El objetivo fue analizar el comportamiento del tiempo de ejecución, speedup y eficiencia al incrementar el número de procesos.

## Ejecución

Desde la raíz del proyecto:

```bat
py run_grid_search.py
```

## Archivos generados

### Resultados del Grid Search

- `grid_search_results_sequential.csv`
- `grid_search_results_parallel_2.csv`
- `grid_search_results_parallel_4.csv`
- `grid_search_results_parallel_6.csv`
- `grid_search_results_parallel_8.csv`

### Información del mejor modelo

- `best_model_info_1.json`
- `best_model_info_2.json`
- `best_model_info_4.json`
- `best_model_info_6.json`
- `best_model_info_8.json`

### Tiempos de ejecución

- `time_grid_search_sequential.txt`
- `time_grid_search_parallel_2.txt`
- `time_grid_search_parallel_4.txt`
- `time_grid_search_parallel_6.txt`
- `time_grid_search_parallel_8.txt`

## Mejor modelo encontrado

El mejor modelo obtenido durante la búsqueda de hiperparámetros fue:

- Modelo: SVM
- Kernel: RBF
- C: 10
- Gamma: scale

Este modelo obtuvo el mejor desempeño entre las configuraciones evaluadas y fue seleccionado como referencia para la etapa de entrenamiento final.

## Objetivo de esta rama

Esta rama contiene exclusivamente la implementación y los resultados asociados a la búsqueda de hiperparámetros y comparación secuencial/paralela del proceso de Grid Search.

La etapa de entrenamiento final y evaluación del modelo seleccionado se encuentra documentada en la rama correspondiente al entrenamiento final.
