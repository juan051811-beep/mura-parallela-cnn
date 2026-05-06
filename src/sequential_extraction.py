import time
import numpy as np
import tensorflow as tf

from data_loader import cargar_rutas
from cnn_model import cargar_cnn


def cargar_imagen(ruta):
    try:
        img = tf.keras.utils.load_img(ruta, target_size=(224, 224))
        img = tf.keras.utils.img_to_array(img)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img
    except Exception:
        print(f"Imagen inválida, se omite: {ruta}")
        return None


def extraer_features_secuencial(dataset_path, batch_size=32):
    rutas, etiquetas = cargar_rutas(dataset_path)

    modelo = cargar_cnn()

    features = []
    labels = []
    imagenes_invalidas = 0

    inicio = time.time()

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

        print(f"Procesadas {min(i + batch_size, len(rutas))}/{len(rutas)} imágenes")

    fin = time.time()

    features = np.vstack(features)
    labels = np.array(labels)

    np.save("features_sequential.npy", features)
    np.save("labels_sequential.npy", labels)

    with open("time_sequential.txt", "w") as f:
        f.write(f"Tiempo secuencial: {fin - inicio:.4f} segundos\n")
        f.write(f"Total imágenes originales: {len(rutas)}\n")
        f.write(f"Imágenes válidas procesadas: {len(labels)}\n")
        f.write(f"Imágenes inválidas omitidas: {imagenes_invalidas}\n")
        f.write(f"Shape features: {features.shape}\n")
        f.write(f"Shape labels: {labels.shape}\n")

    print("\nExtracción secuencial terminada")
    print("Features:", features.shape)
    print("Labels:", labels.shape)
    print("Imágenes inválidas:", imagenes_invalidas)
    print("Tiempo:", fin - inicio, "segundos")


if __name__ == "__main__":
    extraer_features_secuencial("data/MURA-v1.1", batch_size=32)