"""
Extraccion paralela de caracteristicas, la carga de rutas y etiquetas se toma desde data_loader.py y la division del trabajo desde chunks.py.
"""

import time
import multiprocessing
import numpy as np
from PIL import Image

from data_loader import cargar_rutas
from chunks import dividir_chunks

def extraer_caracteristicas_imagen(ruta_imagen, etiqueta):
    #Se extrae caracteristicas basicas de una imagen.
    try:
        imagen = Image.open(ruta_imagen).convert("L")
        arreglo = np.array(imagen)

        caracteristicas = {
            "ruta": ruta_imagen,
            "etiqueta": etiqueta,
            "alto": arreglo.shape[0],
            "ancho": arreglo.shape[1],
            "promedio": float(np.mean(arreglo)),
            "desviacion": float(np.std(arreglo)),
            "minimo": int(np.min(arreglo)),
            "maximo": int(np.max(arreglo))
        }

        return caracteristicas

    except Exception as error:
        return {
            "ruta": ruta_imagen,
            "etiqueta": etiqueta,
            "error": str(error)
        }

def procesar_chunk(chunk):
    """
    con esta función procesamos un bloque de imagenes, cada proceso ejecutara esta funcion con un chunk diferente.
    """
    resultados = []

    rutas = chunk["rutas"]
    etiquetas = chunk["etiquetas"]

    for ruta, etiqueta in zip(rutas, etiquetas):
        caracteristicas = extraer_caracteristicas_imagen(ruta, etiqueta)
        resultados.append(caracteristicas)

    return resultados

def extraccion_paralela(dataset_path, n_procesos):
    """
    se ejecuta la extraccion de caracteristicas.
    """
    rutas, etiquetas = cargar_rutas(dataset_path)
    chunks = dividir_chunks(rutas, etiquetas, n_procesos)

    inicio = time.time()


    fin = time.time()
    tiempo_total = fin - inicio

    return [], tiempo_total


if __name__ == "__main__":
    dataset_path = "data/MURA-v1.1"
    n_procesos = 4

    resultados, tiempo = extraccion_paralela(dataset_path, n_procesos)

    print("Extraccion paralela de caracteristicas")
    print("Procesos utilizados:", n_procesos)
    print("Caracteristicas extraidas:", len(resultados))
    print("Tiempo total:", tiempo, "segundos")
