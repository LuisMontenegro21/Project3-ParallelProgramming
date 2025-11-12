#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include "pgm.h"

#define DEG2RAD 0.017453292519943295f
#define DEGREE_BINS 180
#define R_BINS 200
#define THRESHOLD 120

// Memoria constante: senos y cosenos
__constant__ float d_cos[DEGREE_BINS];
__constant__ float d_sin[DEGREE_BINS];

// KERNEL: versión con memoria COMPARTIDA
__global__ void houghTransformKernelShared(
    unsigned char* img, int width, int height,
    int* acc, int rBins, int degreeBins, float rMax, float rScale)
{
    extern __shared__ int localAcc[];

    int tid = threadIdx.x;
    int gloID = blockIdx.x * blockDim.x + tid;

    // Inicializar acumulador local
    for (int i = tid; i < rBins * degreeBins; i += blockDim.x)
        localAcc[i] = 0;
    __syncthreads();

    if (gloID < width * height) {
        int y = gloID / width;
        int x = gloID % width;

        int xCoord = x - width / 2;
        int yCoord = height / 2 - y;

        if (img[y * width + x] > 250) {
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
                float r = xCoord * d_cos[tIdx] + yCoord * d_sin[tIdx];
                int rIdx = (int)((r + rMax) * rScale);
                if (rIdx >= 0 && rIdx < rBins) {
                    atomicAdd(&localAcc[tIdx * rBins + rIdx], 1);
                }
            }
        }
    }

    __syncthreads();

    for (int i = tid; i < rBins * degreeBins; i += blockDim.x) {
        int val = localAcc[i];
        if (val > 0)
            atomicAdd(&acc[i], val);
    }
}

// Main
int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Uso: %s <imagen.pgm>\n", argv[0]);
        return 1;
    }

    // Leer imagen
    PGMImage* img = readPGM(argv[1]);
    int width = img->width;
    int height = img->height;
    int imgSize = width * height;

    // Parámetros de transformada
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
    int* d_acc;

    cudaMalloc((void**)&d_img, sizeof(unsigned char) * imgSize);
    cudaMalloc((void**)&d_acc, sizeof(int) * degreeBins * rBins);

    cudaMemcpy(d_img, img->data, sizeof(unsigned char) * imgSize, cudaMemcpyHostToDevice);
    cudaMemset(d_acc, 0, sizeof(int) * degreeBins * rBins);

    // Copiar a memoria constante
    cudaMemcpyToSymbol(d_cos, h_cos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_sin, h_sin, sizeof(float) * degreeBins);

    // Medición de tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (imgSize + threads - 1) / threads;
    size_t sharedMemSize = sizeof(int) * degreeBins * rBins;

    cudaEventRecord(start);
    houghTransformKernelShared<<<blocks, threads, sharedMemSize>>>(
        d_img, width, height, d_acc,
        rBins, degreeBins, rMax, rScale);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Tiempo de ejecución (memoria compartida): %.3f ms\n", ms);

    // Copiar resultados al host
    cudaMemcpy(h_acc, d_acc, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // Dibujo de líneas detectadas
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
                        rgb[3 * idx + 0] = 255;
                        rgb[3 * idx + 1] = 0;
                        rgb[3 * idx + 2] = 0;
                    }
                }
            }
        }
    }

    writePPM("output_shared.ppm", rgb, width, height);
    printf("-> Imagen generada: output_shared.ppm\n");

    //  Reporte de tiempos
    mkdir("reporte", 0777);
    FILE* rep = fopen("reporte/tiempos_shared.txt", "a");
    if (rep) {
        fprintf(rep, "----------------------------------------\n");
        fprintf(rep, "Reporte de ejecución: Hough (memoria COMPARTIDA)\n");
        fprintf(rep, "Imagen: %s\n", argv[1]);
        fprintf(rep, "Resolución: %dx%d\n", width, height);
        fprintf(rep, "Tiempo GPU: %.3f ms (%.0f µs, %.6f s)\n",
                ms, ms * 1000.0f, ms / 1000.0f);
        fprintf(rep, "----------------------------------------\n\n");
        fclose(rep);
        printf("-> Reporte guardado en reporte/tiempos_shared.txt\n");
    } else {
        printf("-> No se pudo crear reporte/tiempos_shared.txt\n");
    }

    // Liberar memoria
    cudaFree(d_img);
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
