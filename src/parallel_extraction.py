"""
Extraccion paralela de caracteristicas, la carga de rutas y etiquetas se toma desde data_loader.py y la division del trabajo desde chunks.py.
"""

import time
import multiprocessing

from data_loader import cargar_rutas
from chunks import dividir_chunks


def procesar_chunk(chunk):
    """
    en esta función se procesa un bloque de imagenes, cada chunk contiene una lista de rutas y una lista de etiquetas, más adelante aqui se agregara la extraccion de caracteristicas.
    """
    resultados = []

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
