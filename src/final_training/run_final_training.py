import time
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# Cargar features y labels generados previamente
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


# Entrenar y evaluar el SVM ganador del Grid Search
def entrenar_evaluar(args):
    X_train, X_test, y_train, y_test = args

    modelo = SVC(
        C=10,
        gamma="scale",
        kernel="rbf"
    )

    inicio = time.time()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    fin = time.time()

    return {
        "tiempo": fin - inicio,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }


# Ejecutar experimento usando multiprocessing.Pool
def ejecutar_experimento(n_procesos, X_train, X_test, y_train, y_test):
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

    resultado = resultados[0]
    tiempo_total = fin_total - inicio_total

    if n_procesos == 1:
        archivo_tiempo = "time_training_sequential.txt"
    else:
        archivo_tiempo = f"time_training_parallel_{n_procesos}.txt"

    with open(archivo_tiempo, "w", encoding="utf-8") as f:
        f.write(f"Procesos utilizados: {n_procesos}\n")
        f.write(f"Tiempo total: {tiempo_total:.4f} segundos\n")
        f.write("Paralelismo experimental con multiprocessing.Pool\n")

    return {
        "procesos": n_procesos,
        "tiempo_total": tiempo_total,
        "accuracy": resultado["accuracy"],
        "precision": resultado["precision"],
        "recall": resultado["recall"],
        "f1_score": resultado["f1_score"],
        "classification_report": resultado["classification_report"],
        "confusion_matrix": resultado["confusion_matrix"]
    }


if __name__ == "__main__":
    multiprocessing.freeze_support()

    X_train, X_test, y_train, y_test = cargar_datos()

    procesos = [1, 2, 4, 6, 8]
    resultados = []

    for n in procesos:
        print(f"\nEjecutando entrenamiento con {n} proceso(s)...")
        resultado = ejecutar_experimento(n, X_train, X_test, y_train, y_test)
        resultados.append(resultado)

    # Guardar resumen de métricas
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

    df.to_csv("metrics_summary.csv", index=False)

    # Calcular speedup y eficiencia
    tiempo_secuencial = df[df["procesos"] == 1]["tiempo_total"].values[0]

    df["speedup"] = tiempo_secuencial / df["tiempo_total"]
    df["eficiencia"] = df["speedup"] / df["procesos"]

    df[["procesos", "speedup"]].to_csv("speedup_training.csv", index=False)
    df[["procesos", "eficiencia"]].to_csv("efficiency_training.csv", index=False)

    # Guardar classification report del experimento con 1 proceso
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(resultados[0]["classification_report"])

    # Guardar matriz de confusión del experimento con 1 proceso
    matriz = resultados[0]["confusion_matrix"]

    disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
    disp.plot(cmap="Blues")
    plt.title("Matriz de confusión - SVM final")
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("\nEntrenamiento final terminado.")
    print("Archivos generados:")
    print("metrics_summary.csv")
    print("classification_report.txt")
    print("confusion_matrix.png")
    print("speedup_training.csv")
    print("efficiency_training.csv")
    print("time_training_*.txt")
