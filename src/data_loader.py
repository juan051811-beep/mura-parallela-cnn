import os

def cargar_rutas(dataset_path):
    rutas = []
    etiquetas = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                ruta = os.path.join(root, file)
                rutas.append(ruta)

                ruta_lower = ruta.lower()

                if "positive" in ruta_lower:
                    etiquetas.append(1)
                elif "negative" in ruta_lower:
                    etiquetas.append(0)
                else:
                    etiquetas.append(-1)

    return rutas, etiquetas