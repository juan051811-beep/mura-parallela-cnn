import time
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)


def cargar_datos():
    X = np.load("features_sequential.npy")
    y = np.load("labels_sequential.npy")

    return train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# Entrena un modelo de Regresión Logística sobre un chunk del dataset
def entrenar_modelo_chunk(args):
    X_chunk, y_chunk = args

    modelo = LogisticRegression(
        C=0.1,
        solver="lbfgs",
        max_iter=1000
    )

    modelo.fit(X_chunk, y_chunk)
    return modelo


# Combina las predicciones de todos los modelos por promedio de probabilidades
def predecir_ensemble(modelos, X_test):
    probabilidades = []

    for modelo in modelos:
        probabilidades.append(modelo.predict_proba(X_test))

    promedio = np.mean(probabilidades, axis=0)
    return np.argmax(promedio, axis=1)


def ejecutar_experimento(n_procesos, X_train, X_test, y_train, y_test):
    # Siempre se usan 8 chunks para que todos los experimentos tengan el mismo trabajo
    X_chunks = np.array_split(X_train, 8)
    y_chunks = np.array_split(y_train, 8)

    tareas = list(zip(X_chunks, y_chunks))

    inicio = time.time()

    if n_procesos == 1:
        modelos = [entrenar_modelo_chunk(tarea) for tarea in tareas]
    else:
        with multiprocessing.Pool(processes=n_procesos) as pool:
            modelos = pool.map(entrenar_modelo_chunk, tareas)

    y_pred = predecir_ensemble(modelos, X_test)

    fin = time.time()
    tiempo_total = fin - inicio

    if n_procesos == 1:
        archivo_tiempo = "time_training_sequential.txt"
    else:
        archivo_tiempo = f"time_training_parallel_{n_procesos}.txt"

    with open(archivo_tiempo, "w", encoding="utf-8") as f:
        f.write(f"Procesos utilizados: {n_procesos}\n")
        f.write(f"Tiempo total: {tiempo_total:.4f} segundos\n")
        f.write("Modelo: Regresión Logística en ensemble por chunks\n")

    return {
        "procesos": n_procesos,
        "tiempo_total": tiempo_total,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }


if __name__ == "__main__":
    multiprocessing.freeze_support()

    X_train, X_test, y_train, y_test = cargar_datos()

    procesos = [1, 2, 4, 6, 8]
    resultados = []

    for n in procesos:
        print(f"\nEntrenando con {n} proceso(s)...")
        resultado = ejecutar_experimento(n, X_train, X_test, y_train, y_test)
        resultados.append(resultado)

        print(f"Tiempo: {resultado['tiempo_total']:.2f} s")
        print(f"Accuracy: {resultado['accuracy']:.4f}")
        print(f"F1-score: {resultado['f1_score']:.4f}")

    df = pd.DataFrame([
        {
            "procesos": r["procesos"],
            "tiempo_total": r["tiempo_total"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1_score": r["f1_score"]
        }
        for r in resultados
    ])

    tiempo_secuencial = df[df["procesos"] == 1]["tiempo_total"].values[0]

    df["speedup"] = tiempo_secuencial / df["tiempo_total"]
    df["eficiencia"] = df["speedup"] / df["procesos"]

    df.to_csv("metrics_summary.csv", index=False)
    df[["procesos", "speedup"]].to_csv("speedup_training.csv", index=False)
    df[["procesos", "eficiencia"]].to_csv("efficiency_training.csv", index=False)

    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(resultados[0]["classification_report"])

    matriz = resultados[0]["confusion_matrix"]
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
    disp.plot(cmap="Blues")
    plt.title("Matriz de confusión - Regresión Logística final")
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("\nEntrenamiento final terminado.")
    print("Archivos generados correctamente.")
