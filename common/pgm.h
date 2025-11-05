#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Estructura básica de una imagen PGM
typedef struct {
    int width;
    int height;
    int* data;
} PGMImage;

// Lee una imagen PGM (P5 binario). Retorna puntero a PGMImage con data asignada con malloc.
// Debe liberarse con free(img->data); free(img);
PGMImage* readPGM(const char* filename);

// Escribe una imagen PGM (P5 binario) desde un arreglo de enteros [0,255]
void writePGM(const char* filename, const int* data, int width, int height);

#ifdef __cplusplus
}
#endif
