#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int width;
    int height;
    int* data;
} PGMImage;

PGMImage* readPGM(const char* filename);

void writePGM(const char* filename, const int* data, int width, int height);

#ifdef __cplusplus
}
#endif
