import time
import multiprocessing
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# Carga las características extraídas por la CNN
# y divide el dataset en entrenamiento y prueba (80/20)
def cargar_datos():
    X = np.load("features_sequential.npy")
    y = np.load("labels_sequential.npy")

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# Entrena y evalúa el modelo SVM utilizando
# los mejores hiperparámetros obtenidos en Grid Search
def entrenar_evaluar(args):
    X_train, X_test, y_train, y_test = args

    # Modelo ganador obtenido por Joel
    modelo = SVC(
        C=10,
        gamma="scale",
        kernel="rbf"
    )

    # Medición del tiempo de entrenamiento
    inicio = time.time()

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    fin = time.time()

    # Retornar métricas para el análisis posterior
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "tiempo": fin - inicio,
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

# Entrena y evalúa el modelo SVM utilizando
# los mejores hiperparámetros obtenidos en Grid Search
def entrenar_evaluar(args):
    X_train, X_test, y_train, y_test = args

    # Modelo ganador obtenido por Joel
    modelo = SVC(
        C=10,
        gamma="scale",
        kernel="rbf"
    )

    # Medición del tiempo de entrenamiento
    inicio = time.time()

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    fin = time.time()

    # Retornar métricas para el análisis posterior
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "tiempo": fin - inicio,
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
# Ejecuta el entrenamiento utilizando multiprocessing.Pool
def ejecutar_experimento(n_procesos):
    X_train, X_test, y_train, y_test = cargar_datos()

    # Crear las tareas que ejecutará cada proceso
    tareas = [
        (X_train, X_test, y_train, y_test)
        for _ in range(n_procesos)
    ]

    inicio_total = time.time()

    if n_procesos == 1:
        resultados = [entrenar_evaluar(tareas[0])]
    else:
        with multiprocessing.Pool(processes=n_procesos) as pool:
            resultados = pool.map(entrenar_evaluar, tareas)

    fin_total = time.time()

    # Se toma el primer resultado porque todos usan
    # exactamente el mismo conjunto de datos y parámetros
    mejor_resultado = resultados[0]

    print("\n====================================")
    print(f"Procesos utilizados: {n_procesos}")
    print(f"Tiempo total: {fin_total - inicio_total:.2f} segundos")
    print(f"Accuracy: {mejor_resultado['accuracy']:.4f}")
    print(f"Precision: {mejor_resultado['precision']:.4f}")
    print(f"Recall: {mejor_resultado['recall']:.4f}")
    print(f"F1-score: {mejor_resultado['f1_score']:.4f}")
    print("====================================")

    return {
        "procesos": n_procesos,
        "tiempo_total": fin_total - inicio_total,
        "accuracy": mejor_resultado["accuracy"],
        "precision": mejor_resultado["precision"],
        "recall": mejor_resultado["recall"],
        "f1_score": mejor_resultado["f1_score"]
    }


if __name__ == "__main__":
    multiprocessing.freeze_support()

    procesos = [1, 2, 4, 6, 8]

    resultados = []

    for n in procesos:
        resultados.append(ejecutar_experimento(n))

    df = pd.DataFrame(resultados)

    df.to_csv("metrics_summary.csv", index=False)

    print("\nResumen guardado en metrics_summary.csv")
