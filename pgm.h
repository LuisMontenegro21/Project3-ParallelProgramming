#ifndef PGM_H
#define PGM_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int width;
    int height;
    unsigned char* data;
} PGMImage;

PGMImage* readPGM(const char* filename);
void writePGM(const char* filename, PGMImage* img);
void freePGM(PGMImage* img); // check if reallty needed

#endif