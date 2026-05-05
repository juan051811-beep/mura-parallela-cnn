import pandas as pd  
from data_loader import cargar_rutas 
from chunks import dividir_chunks

rutas, etiquetas = cargar_rutas("data/MURA-v1.1") #cargamos la rita y las etiquetas del dataset

print("Total imágenes:", len(rutas))#mostramos el número total de imágenes
print("Normal:", etiquetas.count(0)) #contamos cuántas son normales
print("Abnormal:", etiquetas.count(1)) #contamos cuántas son anormales
print("Sin etiqueta:", etiquetas.count(-1)) #contamos si hay etiquetas no identificadas


chunks = dividir_chunks(rutas, etiquetas, 4) #dividimos el dataset en 4 partes (solo prueba)

print("\nChunks:") #mostramos los chunks
for i, chunk in enumerate(chunks): #recorremos cada chunk
    print(f"Chunk {i+1}: {len(chunk['rutas'])} imágenes") #tamaño de cada chunk


total_chunks = sum(len(c['rutas']) for c in chunks) #suma total de elementos en todos los chunks

print("\nValidación:")
print("Total original:", len(rutas)) #total original de imágenes
print("Total en chunks:", total_chunks) #total después de dividir

if total_chunks == len(rutas): #verificamos que no se hayan perdido datos
    print("Bien, no se perdieron datos") 
else:
    print("Error") 


df = pd.DataFrame({ #creamos un DataFrame con rutas y etiquetas
    "ruta": rutas,
    "etiqueta": etiquetas
})

df.to_csv("data/metadata.csv", index=False) #guardamos el metadata en archivo CSV

print("\n metadata.csv guardado") 
