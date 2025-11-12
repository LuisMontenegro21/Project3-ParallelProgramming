#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include "pgm.h"

#define DEG2RAD 0.017453292519943295f
#define DEGREE_BINS 180
#define R_BINS 200
#define THRESHOLD 120
#define ITERACIONES 10

// Memoria constante
__constant__ float d_cos[DEGREE_BINS];
__constant__ float d_sin[DEGREE_BINS];

// Kernel: memoria GLOBAL
__global__ void kernelGlobal(unsigned char* img, int width, int height,
                             int* acc, float* cosT, float* sinT,
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
            float r = xCoord * cosT[tIdx] + yCoord * sinT[tIdx];
            int rIdx = (int)((r + rMax) * rScale);
            if (rIdx >= 0 && rIdx < rBins)
                atomicAdd(&acc[tIdx * rBins + rIdx], 1);
        }
    }
}

// Kernel: memoria CONSTANTE
__global__ void kernelConst(unsigned char* img, int width, int height,
                            int* acc, int rBins, int degreeBins,
                            float rMax, float rScale)
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
            if (rIdx >= 0 && rIdx < rBins)
                atomicAdd(&acc[tIdx * rBins + rIdx], 1);
        }
    }
}

// Kernel: memoria COMPARTIDA
__global__ void kernelShared(unsigned char* img, int width, int height,
                             int* acc, int rBins, int degreeBins,
                             float rMax, float rScale)
{
    extern __shared__ int localAcc[];
    int tid = threadIdx.x;
    int gloID = blockIdx.x * blockDim.x + tid;

    // Inicialización
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
                if (rIdx >= 0 && rIdx < rBins)
                    atomicAdd(&localAcc[tIdx * rBins + rIdx], 1);
            }
        }
    }
    __syncthreads();

    // Fusionar local → global
    for (int i = tid; i < rBins * degreeBins; i += blockDim.x) {
        int val = localAcc[i];
        if (val > 0)
            atomicAdd(&acc[i], val);
    }
}

// Función para generar imagen RGB con líneas detectadas
void dibujarLineasColor(const char* nombre_salida, const PGMImage* img,
                        int* h_acc, int width, int height,
                        int rBins, int degreeBins,
                        float rMax, float rScale,
                        unsigned char rColor, unsigned char gColor, unsigned char bColor)
{
    unsigned char* rgb = (unsigned char*)malloc(width * height * 3);

    // Convertir imagen original a RGB base gris
    for (int i = 0; i < width * height; i++) {
        unsigned char gray = img->data[i];
        rgb[3 * i + 0] = gray;
        rgb[3 * i + 1] = gray;
        rgb[3 * i + 2] = gray;
    }

    // Dibujar líneas
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
                        rgb[3 * idx + 0] = rColor;
                        rgb[3 * idx + 1] = gColor;
                        rgb[3 * idx + 2] = bColor;
                    }
                }
            }
        }
    }

    writePPM(nombre_salida, rgb, width, height);
    free(rgb);
}

// Función para ejecutar N veces y promediar
void ejecutarKernelGlobal(FILE* rep, unsigned char* d_img, int* d_acc,
                          float* d_cos, float* d_sin,
                          int width, int height,
                          int rBins, int degreeBins, float rMax, float rScale)
{
    int imgSize = width * height;
    int threads = 256;
    int blocks = (imgSize + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    fprintf(rep, "FASE 1 - Memoria GLOBAL\n");
    for (int i = 0; i < ITERACIONES; i++) {
        cudaMemset(d_acc, 0, sizeof(int) * degreeBins * rBins);
        cudaEventRecord(start);
        kernelGlobal<<<blocks, threads>>>(d_img, width, height, d_acc,
                                          d_cos, d_sin, rBins, degreeBins, rMax, rScale);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        fprintf(rep, "Iteración %02d: %.5f ms\n", i + 1, ms);
    }
    fprintf(rep, "----------------------------------------------\n\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void ejecutarKernelConst(FILE* rep, unsigned char* d_img, int* d_acc,
                         int width, int height,
                         int rBins, int degreeBins, float rMax, float rScale)
{
    int imgSize = width * height;
    int threads = 256;
    int blocks = (imgSize + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    fprintf(rep, "FASE 2 - Memoria GLOBAL + CONSTANTE\n");
    for (int i = 0; i < ITERACIONES; i++) {
        cudaMemset(d_acc, 0, sizeof(int) * degreeBins * rBins);
        cudaEventRecord(start);
        kernelConst<<<blocks, threads>>>(d_img, width, height, d_acc,
                                         rBins, degreeBins, rMax, rScale);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        fprintf(rep, "Iteración %02d: %.5f ms\n", i + 1, ms);
    }
    fprintf(rep, "----------------------------------------------\n\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void ejecutarKernelShared(FILE* rep, unsigned char* d_img, int* d_acc,
                          int width, int height,
                          int rBins, int degreeBins, float rMax, float rScale)
{
    int imgSize = width * height;
    int threads = 256;
    int blocks = (imgSize + threads - 1) / threads;
    size_t sharedSize = sizeof(int) * rBins * degreeBins;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    fprintf(rep, "FASE 3 - Memoria GLOBAL + CONSTANTE + COMPARTIDA\n");
    for (int i = 0; i < ITERACIONES; i++) {
        cudaMemset(d_acc, 0, sizeof(int) * degreeBins * rBins);
        cudaEventRecord(start);
        kernelShared<<<blocks, threads, sharedSize>>>(d_img, width, height,
                                                      d_acc, rBins, degreeBins, rMax, rScale);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        fprintf(rep, "Iteración %02d: %.5f ms\n", i + 1, ms);
    }
    fprintf(rep, "----------------------------------------------\n\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// MAIN
int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Uso: %s <imagen.pgm>\n", argv[0]);
        return 1;
    }

    // Leer imagen
    PGMImage* img = readPGM(argv[1]);
    int width = img->width, height = img->height;
    int imgSize = width * height;
    int degreeBins = DEGREE_BINS, rBins = R_BINS;
    float rMax = sqrtf(width * width + height * height) / 2.0f;
    float rScale = (float)rBins / (2.0f * rMax);

    // Host
    float *h_cos = (float*)malloc(sizeof(float) * degreeBins);
    float *h_sin = (float*)malloc(sizeof(float) * degreeBins);
    for (int i = 0; i < degreeBins; i++) {
        float rad = i * DEG2RAD;
        h_cos[i] = cosf(rad);
        h_sin[i] = sinf(rad);
    }

    // Device
    unsigned char* d_img;
    float *d_cos, *d_sin;
    int* d_acc;
    cudaMalloc(&d_img, sizeof(unsigned char) * imgSize);
    cudaMalloc(&d_cos, sizeof(float) * degreeBins);
    cudaMalloc(&d_sin, sizeof(float) * degreeBins);
    cudaMalloc(&d_acc, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_img, img->data, sizeof(unsigned char) * imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, h_cos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, h_sin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_cos, h_cos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_sin, h_sin, sizeof(float) * degreeBins);

    mkdir("reporte", 0777);
    FILE* rep = fopen("reporte/bitacora_tiempos.txt", "a");
    fprintf(rep, "================ BITÁCORA DE EJECUCIÓN ================\n");
    fprintf(rep, "Imagen: %s\nResolución: %dx%d\nIteraciones por fase: %d\n\n", argv[1], width, height, ITERACIONES);

    // ---- FASE 1: Global ----
    ejecutarKernelGlobal(rep, d_img, d_acc, d_cos, d_sin,
                     width, height, rBins, degreeBins, rMax, rScale);
    printf("Fase 1 completada (GLOBAL)\n");

    int* h_acc_global = (int*)calloc(degreeBins * rBins, sizeof(int));
    cudaMemcpy(h_acc_global, d_acc, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);
    dibujarLineasColor("output_global.ppm", img, h_acc_global,
                    width, height, rBins, degreeBins, rMax, rScale,
                    0, 0, 255); // azul
    free(h_acc_global);
    printf("Imagen generada: output_global.ppm\n");


    // ---- FASE 2: Global + Constante ----
    ejecutarKernelConst(rep, d_img, d_acc,
                    width, height, rBins, degreeBins, rMax, rScale);
    printf("Fase 2 completada (CONSTANTE)\n");

    int* h_acc_const = (int*)calloc(degreeBins * rBins, sizeof(int));
    cudaMemcpy(h_acc_const, d_acc, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);
    dibujarLineasColor("output_const.ppm", img, h_acc_const,
                    width, height, rBins, degreeBins, rMax, rScale,
                    0, 255, 0); // verde
    free(h_acc_const);
    printf("Imagen generada: output_const.ppm\n");


    // ---- FASE 3: Global + Constante + Compartida ----
    ejecutarKernelShared(rep, d_img, d_acc,
                     width, height, rBins, degreeBins, rMax, rScale);
    printf("Fase 3 completada (COMPARTIDA)\n");
    fprintf(rep, "--------------------------------------------------------\n\n");
    fclose(rep);
    printf("Resultados guardados en reporte/bitacora_tiempos.txt\n");

    int* h_acc_shared = (int*)calloc(degreeBins * rBins, sizeof(int));
    cudaMemcpy(h_acc_shared, d_acc, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);
    dibujarLineasColor("output_shared.ppm", img, h_acc_shared,
                    width, height, rBins, degreeBins, rMax, rScale,
                    255, 0, 0); // rojo
    free(h_acc_shared);
    printf("Imagen generada: output_shared.ppm\n");


    // Liberar
    cudaFree(d_img);
    cudaFree(d_cos);
    cudaFree(d_sin);
    cudaFree(d_acc);
    free(h_cos);
    free(h_sin);
    freePGM(img);

    return 0;
}