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
