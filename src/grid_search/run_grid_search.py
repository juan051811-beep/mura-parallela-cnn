import time
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from grid_search_models import obtener_modelos_grid


def cargar_muestra_estratificada():
    X = np.load("features_sequential.npy")
    y = np.load("labels_sequential.npy")

    X_muestra, _, y_muestra, _ = train_test_split(
        X,
        y,
        train_size=0.30,
        random_state=42,
        stratify=y
    )

    return X_muestra, y_muestra


def ejecutar_grid_search(n_jobs):
    X, y = cargar_muestra_estratificada()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    modelos = obtener_modelos_grid()
    resultados = []

    inicio_total = time.time()

    for nombre, config in modelos.items():
        print("\n====================================")
        print(f"Grid Search: {nombre} | n_jobs={n_jobs}")
        print("====================================")

        inicio_modelo = time.time()

        grid = GridSearchCV(
            estimator=config["modelo"],
            param_grid=config["parametros"],
            scoring="f1",
            cv=3,
            n_jobs=n_jobs,
            verbose=2,
            refit=True
        )

        grid.fit(X_train, y_train)

        fin_modelo = time.time()

        mejor_modelo = grid.best_estimator_
        y_pred = mejor_modelo.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        resultados.append({
            "modelo": nombre,
            "n_jobs": n_jobs,
            "mejores_parametros": grid.best_params_,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "tiempo_segundos": fin_modelo - inicio_modelo
        })

    fin_total = time.time()

    df = pd.DataFrame(resultados)

    if n_jobs == 1:
        archivo_resultados = "grid_search_results_sequential.csv"
        archivo_tiempo = "time_grid_search_sequential.txt"
    else:
        archivo_resultados = f"grid_search_results_parallel_{n_jobs}.csv"
        archivo_tiempo = f"time_grid_search_parallel_{n_jobs}.txt"

    df.to_csv(archivo_resultados, index=False)

    mejor = max(resultados, key=lambda r: r["f1_score"])

    with open(archivo_tiempo, "w", encoding="utf-8") as f:
        f.write(f"Tiempo total: {fin_total - inicio_total:.4f} segundos\n")
        f.write(f"n_jobs: {n_jobs}\n")
        f.write("Muestra usada: 30% estratificado del dataset\n")
        f.write(f"Mejor modelo: {mejor['modelo']}\n")
        f.write(f"Mejor F1-score: {mejor['f1_score']:.4f}\n")
        f.write(f"Mejor accuracy: {mejor['accuracy']:.4f}\n")
        f.write(f"Mejor precision: {mejor['precision']:.4f}\n")
        f.write(f"Mejor recall: {mejor['recall']:.4f}\n")
        f.write(f"Mejores parametros: {mejor['mejores_parametros']}\n")

    with open(f"best_model_info_{n_jobs}.json", "w", encoding="utf-8") as f:
        json.dump(mejor, f, indent=4)

    print("\nGrid Search terminado")
    print(df)
    print("\nMejor modelo:")
    print(mejor)


def ejecutar_todas_las_pruebas():
    for n_jobs in [1, 2, 4, 6, 8]:
        ejecutar_grid_search(n_jobs)


if __name__ == "__main__":
    ejecutar_todas_las_pruebas()
