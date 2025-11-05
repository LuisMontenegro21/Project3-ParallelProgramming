#include "pgm.h"
#include <ctype.h>


// TODO make sure this is actually working accodringly for pgm handling
PGMImage* readPGM(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    char format[3];
    if (fscanf(file, "%2s", format) != 1 || strcmp(format, "P5") != 0) {
        fprintf(stderr, "Unsupported PGM format or invalid file\n");
        fclose(file);
        return NULL;
    }

    int width, height, maxval;
    // Skip comments
    int c;
    while ((c = fgetc(file)) == '#') {
        while (fgetc(file) != '\n');
    }
    ungetc(c, file);

    if (fscanf(file, "%d %d %d", &width, &height, &maxval) != 3 || maxval != 255) {
        fprintf(stderr, "Invalid PGM header\n");
        fclose(file);
        return NULL;
    }

    fgetc(file); // Consume the newline after maxval

    PGMImage* img = (PGMImage*)malloc(sizeof(PGMImage));
    img->width = width;
    img->height = height;
    img->data = (unsigned char*)malloc(width * height);

    if (fread(img->data, 1, width * height, file) != width * height) {
        fprintf(stderr, "Error reading pixel data\n");
        free(img->data);
        free(img);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return img;
}

void writePGM(const char* filename, PGMImage* img) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file for writing");
        return;
    }

    fprintf(file, "P5\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, img->width * img->height, file);

    fclose(file);
}