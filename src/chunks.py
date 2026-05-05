def dividir_chunks(rutas, etiquetas, n_procesos):
    chunks = []

    total = len(rutas)
    tamaño_base = total // n_procesos
    sobrante = total % n_procesos

    inicio = 0

    for i in range(n_procesos):
        extra = 1 if i < sobrante else 0
        fin = inicio + tamaño_base + extra

        chunk = {
            "rutas": rutas[inicio:fin],
            "etiquetas": etiquetas[inicio:fin]
        }

        chunks.append(chunk)
        inicio = fin

    return chunks