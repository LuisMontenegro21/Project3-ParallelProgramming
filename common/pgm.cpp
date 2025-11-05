#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Estructura básica de una imagen PGM
typedef struct {
    int width;
    int height;
    int* data;
} PGMImage;

// Función para leer una imagen PGM en escala de grises
PGMImage* readPGM(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: no se pudo abrir el archivo %s\n", filename);
        exit(1);
    }

    char magic[3];
    if (!fgets(magic, sizeof(magic), fp)) {
        fprintf(stderr, "Error: formato incorrecto en %s\n", filename);
        exit(1);
    }

    if (strcmp(magic, "P5") != 0 && strcmp(magic, "P5\n") != 0) {
        fprintf(stderr, "Error: el formato de imagen no es P5 (PGM binario)\n");
        exit(1);
    }

    int width, height, maxval;
    int c;
    // Saltar comentarios (#)
    c = fgetc(fp);
    while (c == '#') {
        while (fgetc(fp) != '\n');
        c = fgetc(fp);
    }
    ungetc(c, fp);

    // Leer ancho, alto y valor máximo
    fscanf(fp, "%d %d", &width, &height);
    fscanf(fp, "%d", &maxval);
    fgetc(fp); // Salto de línea después de maxval

    // Reservar memoria y leer datos
    int* data = (int*)malloc(width * height * sizeof(int));
    for (int i = 0; i < width * height; i++) {
        unsigned char pixel;
        fread(&pixel, sizeof(unsigned char), 1, fp);
        data[i] = (int)pixel;
    }
    fclose(fp);

    PGMImage* img = (PGMImage*)malloc(sizeof(PGMImage));
    img->width = width;
    img->height = height;
    img->data = data;
    return img;
}

// Función para guardar una imagen PGM en escala de grises
void writePGM(const char* filename, int* data, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: no se pudo escribir en %s\n", filename);
        exit(1);
    }

    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; i++) {
        unsigned char pixel = (unsigned char)(data[i] > 255 ? 255 : (data[i] < 0 ? 0 : data[i]));
        fwrite(&pixel, sizeof(unsigned char), 1, fp);
    }
    fclose(fp);
}
