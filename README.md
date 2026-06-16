# Extracción paralela de características CNN

Esta rama contiene la extracción de características de radiografías óseas utilizando la arquitectura MobileNetV2.

Se implementaron dos versiones:

- Extracción secuencial
- Extracción paralela usando multiprocessing

## Resultados

| Procesos | Tiempo | Speedup | Eficiencia |
|---|---:|---:|---:|
| 1 | 3160.93 s | 1.00 | 100% |
| 2 | 484.75 s | 6.52 | 326% |
| 4 | 408.95 s | 7.73 | 193.25% |
| 6 | 466.29 s | 6.78 | 113% |
| 8 | 479.39 s | 6.59 | 82.38% |

La mejor configuración fue con 4 procesos, obteniendo un tiempo de 408.95 segundos.
