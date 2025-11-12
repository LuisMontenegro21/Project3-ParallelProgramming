# Project3-ParallelProgramming

Transformada de Hough en CUDA con tres variantes de memoria y medición de tiempos.

- Global: tablas trigonométricas en memoria global (device) y acumulador global
- Constante: tablas trigonométricas en `__constant__` (solo lectura, caché de difusión)
- Compartida: lectura de píxeles en bloques y acumulación local por bloque en shared (se reduce al acumulador global)

Cada variante genera una imagen PPM con las líneas detectadas sobre la imagen original y un archivo de tiempos en la carpeta `reporte/`.

Imágenes de salida (color, PPM):
- `output_global.ppm`
- `output_const.ppm`
- `output_shared.ppm`

Reportes de tiempo (texto):
- `reporte/tiempos_global.txt`
- `reporte/tiempos_const.txt`
- `reporte/tiempos_shared.txt`

Entrada: imagen `PGM` en escala de grises (P5), por ejemplo `runway.pgm`.

---

## Requisitos

- NVIDIA GPU o CUDA
- Algun compilador para c/c++ (se recomienda WSL)

Opcional para visualizar: Python con `matplotlib` y `numpy`.

---

## Compilación

Desde la raíz del proyecto:

```bash
make           # compila las 3 variantes: houghGlobal, houghConst, houghShared
make global    # solo memoria global
make const     # solo memoria constante
make shared    # solo memoria compartida
make legacy    # una version completa usando los 3
make clean     # limpia binarios y objetos
```

Esto genera los ejecutables:
- `houghGlobal`
- `houghConst`
- `houghShared`
- `hough_legacy`

---

## Ejecución

Ejemplos (WSL):

```bash
./houghGlobal runway.pgm
./houghConst  runway.pgm
./houghShared runway.pgm
./hough_legacy runway.pgm
```

Cada ejecución:
- Imprime el tiempo del kernel medido con CUDA events
- Genera `output_*.ppm` con las líneas en rojo dibujadas sobre la imagen original
- Registra el tiempo en `reporte/tiempos_*.txt`

En el caso del legacy, hace 10 iteraciones de las diferentes fases y hace los reportes que se guardan en `reporte/bitacora_tiempos.txt` 

---

## Visualizar y comparar outputs

Tenemos el `convert_pgm.py` que abre tanto PGM (grises) como PPM (color).

Instalación:

```bash
pip install matplotlib numpy
```

Uso:

Ver una o varias imágenes y compararlas lado a lado:

```bash
python convert_pgm.py output_global.ppm output_const.ppm output_shared.ppm
```

Guardar la comparación sin abrir ventana:

```bash
python convert_pgm.py output_global.ppm output_const.ppm output_shared.ppm --no-show --save resultados/comparacion.png
```