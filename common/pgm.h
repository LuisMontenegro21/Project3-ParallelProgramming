#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int width;
    int height;
    unsigned char* data;
} PGMImage;

PGMImage* readPGM(const char* filename);

void writePGM(const char* filename, const int* data, int width, int height);
void freePGM(PGMImage* img); // check if reallty needed

#ifdef __cplusplus
}
#endif
