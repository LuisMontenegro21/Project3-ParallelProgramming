#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include "pgm.h"

#define DEG2RAD 0.017453292519943295f
#define DEGREE_BINS 180
#define R_BINS 200
#define THRESHOLD 120

// KERNEL: versión con memoria GLOBAL
__global__ void houghTransformKernel(
    unsigned char* img, int width, int height,
    int* acc, float* d_cos, float* d_sin,
    int rBins, int degreeBins, float rMax, float rScale)
{
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= width * height) return;

    int y = gloID / width;
    int x = gloID % width;

    int xCoord = x - width / 2;
    int yCoord = height / 2 - y;

    if (img[y * width + x] > 250) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_cos[tIdx] + yCoord * d_sin[tIdx];
            int rIdx = (int)((r + rMax) * rScale);
            if (rIdx >= 0 && rIdx < rBins) {
                atomicAdd(&acc[tIdx * rBins + rIdx], 1);
            }
        }
    }
}

// Main
int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Uso: %s <imagen.pgm>\n", argv[0]);
        return 1;
    }

    // Leer imagen de entrada
    PGMImage* img = readPGM(argv[1]);
    int width = img->width;
    int height = img->height;
    int imgSize = width * height;

    // Definir parámetros de la transformada
    int degreeBins = DEGREE_BINS;
    int rBins = R_BINS;
    float rMax = sqrtf(width * width + height * height) / 2.0f;
    float rScale = (float)rBins / (2.0f * rMax);

    // Memoria en host
    float* h_cos = (float*)malloc(sizeof(float) * degreeBins);
    float* h_sin = (float*)malloc(sizeof(float) * degreeBins);
    for (int i = 0; i < degreeBins; i++) {
        float rad = i * DEG2RAD;
        h_cos[i] = cosf(rad);
        h_sin[i] = sinf(rad);
    }

    int* h_acc = (int*)calloc(degreeBins * rBins, sizeof(int));

    // Memoria en device
    unsigned char* d_img;
    float *d_cos, *d_sin;
    int* d_acc;

    cudaMalloc((void**)&d_img, sizeof(unsigned char) * imgSize);
    cudaMalloc((void**)&d_cos, sizeof(float) * degreeBins);
    cudaMalloc((void**)&d_sin, sizeof(float) * degreeBins);
    cudaMalloc((void**)&d_acc, sizeof(int) * degreeBins * rBins);

    cudaMemcpy(d_img, img->data, sizeof(unsigned char) * imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, h_cos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, h_sin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemset(d_acc, 0, sizeof(int) * degreeBins * rBins);

    // Medición de tiempo

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (imgSize + threads - 1) / threads;

    cudaEventRecord(start);
    houghTransformKernel<<<blocks, threads>>>(
        d_img, width, height,
        d_acc, d_cos, d_sin,
        rBins, degreeBins, rMax, rScale);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Tiempo de ejecución (memoria global): %.3f ms\n", ms);

    // Copiar resultados al host
    cudaMemcpy(h_acc, d_acc, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // Dibujo de líneas detectadas sobre la imagen
    unsigned char* rgb = (unsigned char*)malloc(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        rgb[3 * i + 0] = img->data[i];
        rgb[3 * i + 1] = img->data[i];
        rgb[3 * i + 2] = img->data[i];
    }

    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
        for (int rIdx = 0; rIdx < rBins; rIdx++) {
            int val = h_acc[tIdx * rBins + rIdx];
            if (val > THRESHOLD) {
                float theta = tIdx * DEG2RAD;
                float r = rIdx / rScale - rMax;
                for (int x = 0; x < width; x++) {
                    int y = (int)((r - ((x - width / 2) * cosf(theta))) / sinf(theta) + height / 2);
                    if (y >= 0 && y < height) {
                        int idx = y * width + x;
                        rgb[3 * idx + 0] = 255; // rojo
                        rgb[3 * idx + 1] = 0;
                        rgb[3 * idx + 2] = 0;
                    }
                }
            }
        }
    }

    writePPM("output_global.ppm", rgb, width, height);
    printf("-> Imagen generada: output_global.ppm\n");

    // Reporte de tiempos
    mkdir("reporte", 0777);
    FILE* rep = fopen("reporte/tiempos_global.txt", "a");
    if (rep) {
        fprintf(rep, "----------------------------------------\n");
        fprintf(rep, "Reporte de ejecución: Hough (memoria GLOBAL)\n");
        fprintf(rep, "Imagen: %s\n", argv[1]);
        fprintf(rep, "Resolución: %dx%d\n", width, height);
        fprintf(rep, "Tiempo GPU: %.3f ms (%.0f µs, %.6f s)\n",
                ms, ms * 1000.0f, ms / 1000.0f);
        fprintf(rep, "----------------------------------------\n\n");
        fclose(rep);
        printf("-> Reporte guardado en reporte/tiempos_global.txt\n");
    } else {
        printf("-> No se pudo crear reporte/tiempos_global.txt\n");
    }

    // Liberar memoria
    cudaFree(d_img);
    cudaFree(d_cos);
    cudaFree(d_sin);
    cudaFree(d_acc);
    free(h_cos);
    free(h_sin);
    free(h_acc);
    free(rgb);
    freePGM(img);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
