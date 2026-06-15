import time
import json
import multiprocessing
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from grid_search_models import crear_modelo, obtener_parametros_grid


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


def evaluar_parametros(args):
    nombre_modelo, parametros, X_train, y_train = args

    modelo = crear_modelo(nombre_modelo, parametros)

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        modelo,
        X_train,
        y_train,
        scoring="f1",
        cv=cv,
        n_jobs=1
    )

    return {
        "modelo": nombre_modelo,
        "parametros": parametros,
        "f1_cv_promedio": float(scores.mean())
    }


def grid_search_multiprocessing(nombre_modelo, grid_parametros, X_train, y_train, n_procesos):
    combinaciones = list(ParameterGrid(grid_parametros))

    tareas = [
        (nombre_modelo, parametros, X_train, y_train)
        for parametros in combinaciones
    ]

    if n_procesos == 1:
        resultados = [evaluar_parametros(tarea) for tarea in tareas]
    else:
        with multiprocessing.Pool(processes=n_procesos) as pool:
            resultados = pool.map(evaluar_parametros, tareas)

    mejor_resultado = max(resultados, key=lambda r: r["f1_cv_promedio"])

    return mejor_resultado


def ejecutar_grid_search(n_procesos):
    X, y = cargar_muestra_estratificada()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    grids = obtener_parametros_grid()
    resultados_finales = []

    inicio_total = time.time()

    for nombre_modelo, grid_parametros in grids.items():
        print("\n====================================")
        print(f"Grid Search con multiprocessing: {nombre_modelo}")
        print(f"Procesos utilizados: {n_procesos}")
        print("====================================")

        inicio_modelo = time.time()

        mejor = grid_search_multiprocessing(
            nombre_modelo,
            grid_parametros,
            X_train,
            y_train,
            n_procesos
        )

        mejor_modelo = crear_modelo(nombre_modelo, mejor["parametros"])
        mejor_modelo.fit(X_train, y_train)

        y_pred = mejor_modelo.predict(X_test)

        fin_modelo = time.time()

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        resultados_finales.append({
            "modelo": nombre_modelo,
            "procesos": n_procesos,
            "mejores_parametros": mejor["parametros"],
            "f1_cv_promedio": mejor["f1_cv_promedio"],
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "tiempo_segundos": float(fin_modelo - inicio_modelo)
        })

    fin_total = time.time()

    df = pd.DataFrame(resultados_finales)

    if n_procesos == 1:
        archivo_resultados = "grid_search_results_sequential.csv"
        archivo_tiempo = "time_grid_search_sequential.txt"
    else:
        archivo_resultados = f"grid_search_results_parallel_{n_procesos}.csv"
        archivo_tiempo = f"time_grid_search_parallel_{n_procesos}.txt"

    df.to_csv(archivo_resultados, index=False)

    mejor_global = max(resultados_finales, key=lambda r: r["f1_score"])

    with open(archivo_tiempo, "w", encoding="utf-8") as f:
        f.write(f"Tiempo total: {fin_total - inicio_total:.4f} segundos\n")
        f.write(f"Procesos utilizados: {n_procesos}\n")
        f.write("Paralelismo implementado con multiprocessing.Pool\n")
        f.write("Muestra usada: 30% estratificado del dataset\n")
        f.write(f"Mejor modelo: {mejor_global['modelo']}\n")
        f.write(f"Mejor F1-score: {mejor_global['f1_score']:.4f}\n")
        f.write(f"Mejor accuracy: {mejor_global['accuracy']:.4f}\n")
        f.write(f"Mejor precision: {mejor_global['precision']:.4f}\n")
        f.write(f"Mejor recall: {mejor_global['recall']:.4f}\n")
        f.write(f"Mejores parametros: {mejor_global['mejores_parametros']}\n")

    with open(f"best_model_info_{n_procesos}.json", "w", encoding="utf-8") as f:
        json.dump(mejor_global, f, indent=4)

    print("\nGrid Search terminado")
    print(df)
    print("\nMejor modelo global:")
    print(mejor_global)


def ejecutar_todas_las_pruebas():
    procesos = [1, 2, 4, 6, 8]

    for n_procesos in procesos:
        ejecutar_grid_search(n_procesos)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    ejecutar_todas_las_pruebas()
