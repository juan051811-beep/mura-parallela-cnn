import os

def cargar_rutas(dataset_path): #funcionque obtiene rutas de imágenes y sus etiquetas
    rutas = [] #para guardar las rutas de las imágenes
    etiquetas = []  #para guardar las etiquetas (0 o 1)

    for root, _, files in os.walk(dataset_path): #recorremos todas las carpetas del dataset
        for file in files: #recorremos todos los archivos dentro de cada carpeta
            if file.lower().endswith((".png", ".jpg", ".jpeg")): #filtramos solo archivos de imagen
                ruta = os.path.join(root, file) #construimos la ruta completa de la imagen
                rutas.append(ruta) #guardamosla ruta en la lista

                ruta_lower = ruta.lower() #convertimos la ruta a minúsculas para comparar

                if "positive" in ruta_lower:  #si la ruta contiene "positive"
                    etiquetas.append(1)  #se asigna etiqueta 1 (anormal)
                elif "negative" in ruta_lower:  #si contiene "negative"
                    etiquetas.append(0)  #se asigna etiqueta 0 (normal)
                else:
                    etiquetas.append(-1)  #si no se identifica, se marca como -1

    return rutas, etiquetas  #devolvemos listas de rutas y etiquetas
