def dividir_chunks(rutas, etiquetas, n_procesos):  #funcion que divide los datos en N partes según los procesos
    chunks = []  #donde guardamos los chunks

    total = len(rutas) #total de imágenes
    tamaño_base = total // n_procesos #taaño base de cada chunk 
    sobrante = total % n_procesos #elementos que no se reparten de forma exacta

    inicio = 0  #indice inicial para comenzar a dividir

    for i in range(n_procesos): #creamos tantos chunks como procesos

        extra = 1 if i < sobrante else 0 #se reparte 1 extra a los primeros chunks si hay sobrantes
        fin = inicio + tamaño_base + extra #idice final del chunk actual

        chunk = {
            "rutas": rutas[inicio:fin], #sublista de rutas para este chunk
            "etiquetas": etiquetas[inicio:fin] #sblista de etiquetas correspondiente
        }

        chunks.append(chunk) #agregamos el chunk a la lista
        inicio = fin  #actualizamos el inicio para el siguiente chunk

    return chunks #devolvemos la lista completa de chunks
