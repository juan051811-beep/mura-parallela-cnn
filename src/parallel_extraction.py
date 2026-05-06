import time
import multiprocessing
import numpy as np
import tensorflow as tf

from data_loader import cargar_rutas
from chunks import dividir_chunks
from cnn_model import cargar_cnn


def cargar_imagen(ruta):
    try:
        img = tf.keras.utils.load_img(ruta, target_size=(224, 224))
        img = tf.keras.utils.img_to_array(img)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img
    except Exception:
        print(f"Imagen invalida, se omite: {ruta}")
        return None


def procesar_chunk_cnn(chunk):
    #con esta funcion procesamos un bloque de imagenes usando MobileNetV2.
    modelo = cargar_cnn()

    rutas = chunk["rutas"]
    etiquetas = chunk["etiquetas"]

    batch_size = 32
    features = []
    labels = []
    imagenes_invalidas = 0

    for i in range(0, len(rutas), batch_size):
        batch_rutas = rutas[i:i + batch_size]
        batch_labels = etiquetas[i:i + batch_size]

        imagenes = []
        labels_validos = []

        for ruta, label in zip(batch_rutas, batch_labels):
            img = cargar_imagen(ruta)

            if img is not None:
                imagenes.append(img)
                labels_validos.append(label)
            else:
                imagenes_invalidas += 1

        if len(imagenes) == 0:
            continue

        imagenes = np.array(imagenes)
        batch_features = modelo.predict(imagenes, verbose=0)

        features.append(batch_features)
        labels.extend(labels_validos)

    if len(features) > 0:
        features = np.vstack(features)
    else:
        features = np.empty((0, 1280))

    labels = np.array(labels)

    return features, labels, imagenes_invalidas


def extraer_features_paralelo(dataset_path, n_procesos=4):
    #extraems caracteristicas CNN en paralelo dividiendo el dataset en chunks.
    rutas, etiquetas = cargar_rutas(dataset_path)
    chunks = dividir_chunks(rutas, etiquetas, n_procesos)

    inicio = time.time()

    with multiprocessing.Pool(processes=n_procesos) as pool:
        resultados = pool.map(procesar_chunk_cnn, chunks)

    fin = time.time()
    tiempo_total = fin - inicio

    features_totales = []
    labels_totales = []
    invalidas_totales = 0

    for features, labels, invalidas in resultados:
        features_totales.append(features)
        labels_totales.append(labels)
        invalidas_totales += invalidas

    features_totales = np.vstack(features_totales)
    labels_totales = np.concatenate(labels_totales)

    np.save(f"features_parallel_{n_procesos}.npy", features_totales)
    np.save(f"labels_parallel_{n_procesos}.npy", labels_totales)

    with open(f"time_parallel_{n_procesos}.txt", "w") as f:
        f.write(f"Tiempo paralelo: {tiempo_total:.4f} segundos\n")
        f.write(f"Procesos utilizados: {n_procesos}\n")
        f.write(f"Total imagenes originales: {len(rutas)}\n")
        f.write(f"Imagenes validas procesadas: {len(labels_totales)}\n")
        f.write(f"Imagenes invalidas omitidas: {invalidas_totales}\n")
        f.write(f"Shape features: {features_totales.shape}\n")
        f.write(f"Shape labels: {labels_totales.shape}\n")

    print("\nExtraccion paralela terminada")
    print("Procesos utilizados:", n_procesos)
    print("Features:", features_totales.shape)
    print("Labels:", labels_totales.shape)
    print("Imagenes invalidas:", invalidas_totales)
    print("Tiempo:", tiempo_total, "segundos")
    print("Archivo de tiempo:", f"time_parallel_{n_procesos}.txt")

    return tiempo_total


def ejecutar_pruebas_paralelas(dataset_path):
    #Ecutamos la extraccion paralela con diferente numero de procesos.
    
    procesos_prueba = [2, 4, 6, 8]
    tiempos = []

    for n_procesos in procesos_prueba:
        print("Ejecutando con", n_procesos, "procesos")

        tiempo = extraer_features_paralelo(dataset_path, n_procesos)
        tiempos.append({
            "procesos": n_procesos,
            "tiempo_segundos": tiempo
        })

    with open("time_parallel_summary.txt", "w") as f:
        f.write("Resumen de tiempos paralelos\n")

        for resultado in tiempos:
            f.write(
                f"Procesos: {resultado['procesos']} | "
                f"Tiempo: {resultado['tiempo_segundos']:.4f} segundos\n"
            )

    print("\nResumen de tiempos paralelos")
    for resultado in tiempos:
        print(
            "Procesos:",
            resultado["procesos"],
            "| Tiempo:",
            round(resultado["tiempo_segundos"], 4),
            "segundos"
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    ejecutar_pruebas_paralelas("data/MURA-v1.1")
