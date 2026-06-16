# MURA Paralela CNN

Proyecto de Cómputo Paralelo para la clasificación de radiografías óseas utilizando extracción de características mediante CNN, búsqueda de hiperparámetros y entrenamiento de modelos de aprendizaje automático con técnicas de paralelización usando multiprocessing.

## Entrenamiento final

La etapa final utiliza las características extraídas previamente mediante CNN almacenadas en:

- `features_sequential.npy`
- `labels_sequential.npy`

Se implementaron dos scripts principales:

- `src/final_training/run_final_training.py`: entrenamiento paralelo experimental con Regresión Logística usando `multiprocessing.Pool`.
- `src/final_training/run_final_training_lr_experimental.py`: evaluación final de Regresión Logística con los hiperparámetros obtenidos por Grid Search.

### Ejecución

Desde la raíz del proyecto:

```bash
py src\final_training\run_final_training.py
py src\final_training\run_final_training_lr_experimental.py
```
## Resultados finales

Modelo: Logistic Regression

- Dataset completo: 40,005 imágenes
- Entrenamiento: 32,004 imágenes
- Prueba: 8,001 imágenes

### Métricas

- Accuracy: 75.62%
- Precision: 74.48%
- Recall: 61.66%
- F1-score: 67.47%
