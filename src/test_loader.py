import pandas as pd
from data_loader import cargar_rutas
from chunks import dividir_chunks

rutas, etiquetas = cargar_rutas("data/MURA-v1.1")

print("Total imágenes:", len(rutas))
print("Normal:", etiquetas.count(0))
print("Abnormal:", etiquetas.count(1))
print("Sin etiqueta:", etiquetas.count(-1))


chunks = dividir_chunks(rutas, etiquetas, 4)

print("\nChunks:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {len(chunk['rutas'])} imágenes")


total_chunks = sum(len(c['rutas']) for c in chunks)

print("\nValidación:")
print("Total original:", len(rutas))
print("Total en chunks:", total_chunks)

if total_chunks == len(rutas):
    print("Bien, no se perdieron datos")
else:
    print("Error")



df = pd.DataFrame({
    "ruta": rutas,
    "etiqueta": etiquetas
})

df.to_csv("data/metadata.csv", index=False)

print("\n metadata.csv guardado")