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
#define TILE_ANGULOS 32

// ----------------------
// Memoria constante
// ----------------------
__constant__ float d_cos[DEGREE_BINS];
__constant__ float d_sin[DEGREE_BINS];

// ----------------------
// Kernel: memoria GLOBAL
// ----------------------
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

    if (img[y * width + x] > 80) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * cosT[tIdx] + yCoord * sinT[tIdx];
            int rIdx = (int)((r + rMax) * rScale);
            if (rIdx >= 0 && rIdx < rBins)
                atomicAdd(&acc[tIdx * rBins + rIdx], 1);
        }
    }
}

// ----------------------
// Kernel: memoria CONST
// ----------------------
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

    if (img[y * width + x] > 80) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_cos[tIdx] + yCoord * d_sin[tIdx];
            int rIdx = (int)((r + rMax) * rScale);
            if (rIdx >= 0 && rIdx < rBins)
                atomicAdd(&acc[tIdx * rBins + rIdx], 1);
        }
    }
}

// ----------------------
// Kernel: memoria SHARED
// ----------------------
__global__ void kernelShared(unsigned char* img, int width, int height,
                             int* acc, int rBins, int degreeBins,
                             float rMax, float rScale)
{
    extern __shared__ int localAcc[];

    int tid = threadIdx.x;
    int gloID = blockIdx.x * blockDim.x + tid;

    // bloque procesa ángulos desde:
    int angStart = blockIdx.y * TILE_ANGULOS;
    int angEnd   = min(angStart + TILE_ANGULOS, degreeBins);
    int angCount = angEnd - angStart;

    // Inicializar solo el TILE correspondiente (mucho más pequeño)
    for (int i = tid; i < angCount * rBins; i += blockDim.x)
        localAcc[i] = 0;
    __syncthreads();

    if (gloID < width * height) {
        int y = gloID / width;
        int x = gloID % width;
        int xCoord = x - width / 2;
        int yCoord = height / 2 - y;

        if (img[y * width + x] > 80) {
            for (int tIdx = 0; tIdx < angCount; tIdx++) {

                float theta = (angStart + tIdx) * DEG2RAD;
                float r = xCoord * cosf(theta) + yCoord * sinf(theta);

                int rIdx = (int)((r + rMax) * rScale);

                if (rIdx >= 0 && rIdx < rBins) {
                    int localIndex = tIdx * rBins + rIdx;
                    atomicAdd(&localAcc[localIndex], 1);
                }
            }
        }
    }
    __syncthreads();

    // Guardar a acumulador global
    for (int i = tid; i < angCount * rBins; i += blockDim.x) {
        int val = localAcc[i];
        if (val > 0) {
            int globalIndex = (angStart * rBins) + i;
            atomicAdd(&acc[globalIndex], val);
        }
    }
}

// ===================================================
//  GENERAR LA TRANSFORMADA HOUGH A COLOR
// ===================================================
void guardarTransformadaHough(const char* nombre_salida, int* acc,
                              int rBins, int degreeBins)
{
    int maxVal = 0;
    for (int i = 0; i < rBins * degreeBins; i++)
        if (acc[i] > maxVal) maxVal = acc[i];

    unsigned char* img = (unsigned char*)malloc(rBins * degreeBins * 3);

    for (int t = 0; t < degreeBins; t++) {
        for (int r = 0; r < rBins; r++) {
            int val = acc[t * rBins + r];
            float norm = (float)val / maxVal;

            unsigned char red   = (unsigned char)(255 * powf(norm, 0.5f));
            unsigned char green = (unsigned char)(255 * norm * 0.25f);
            unsigned char blue  = (unsigned char)(255 * (1.0f - norm));

            int idx = (t * rBins + r) * 3;
            img[idx + 0] = red;
            img[idx + 1] = green;
            img[idx + 2] = blue;
        }
    }

    writePPM(nombre_salida, img, rBins, degreeBins);
    free(img);
}

// ===================================================
//  DIBUJAR LÍNEAS SOBRE IMAGEN ORIGINAL (RGB)
// ===================================================
void dibujarLineasColor(const char* nombre_salida, const PGMImage* img,
                        int* h_acc, int width, int height,
                        int rBins, int degreeBins,
                        float rMax, float rScale,
                        unsigned char rColor, unsigned char gColor, unsigned char bColor)
{
    unsigned char* rgb = (unsigned char*)malloc(width * height * 3);

    // Fondo gris
    for (int i = 0; i < width * height; i++) {
        unsigned char g = img->data[i];
        rgb[3*i+0] = g;
        rgb[3*i+1] = g;
        rgb[3*i+2] = g;
    }

    // Líneas
    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
        for (int rIdx = 0; rIdx < rBins; rIdx++) {
            if (h_acc[tIdx * rBins + rIdx] > THRESHOLD) {
                float theta = tIdx * DEG2RAD;
                float r = rIdx / rScale - rMax;

                for (int x = 0; x < width; x++) {
                    int y = (int)((r - ((x - width/2)*cosf(theta))) / sinf(theta) + height/2);
                    if (y >= 0 && y < height) {
                        int idx = y * width + x;
                        rgb[3*idx+0] = rColor;
                        rgb[3*idx+1] = gColor;
                        rgb[3*idx+2] = bColor;
                    }
                }
            }
        }
    }

    writePPM(nombre_salida, rgb, width, height);
    free(rgb);
}

// ===================================================
//   EJECUTAR KERNELS (10 mediciones)
// ===================================================

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
        cudaMemset(d_acc, 0, sizeof(int)*rBins*degreeBins);

        cudaEventRecord(start);
        kernelGlobal<<<blocks, threads>>>(d_img, width, height, d_acc,
                                          d_cos, d_sin, rBins, degreeBins, rMax, rScale);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        fprintf(rep, "Iteración %02d: %.5f ms\n", i+1, ms);
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

    fprintf(rep, "FASE 2 - GLOBAL + CONSTANTE\n");

    for (int i = 0; i < ITERACIONES; i++) {
        cudaMemset(d_acc, 0, sizeof(int)*rBins*degreeBins);

        cudaEventRecord(start);
        kernelConst<<<blocks, threads>>>(d_img, width, height, d_acc,
                                         rBins, degreeBins, rMax, rScale);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        fprintf(rep, "Iteración %02d: %.5f ms\n", i+1, ms);
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
    dim3 blockDim(256);
    dim3 gridDim((imgSize + 255) / 256, (degreeBins + TILE_ANGULOS - 1) / TILE_ANGULOS);

    size_t sharedSize = TILE_ANGULOS * rBins * sizeof(int);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    fprintf(rep, "FASE 3 - GLOBAL + CONSTANTE + SHARED\n");

    for (int i = 0; i < ITERACIONES; i++) {
        cudaMemset(d_acc, 0, sizeof(int)*rBins*degreeBins);

        cudaEventRecord(start);
        kernelShared<<<gridDim, blockDim, sharedSize>>>(d_img, width, height,
                                                      d_acc, rBins, degreeBins, rMax, rScale);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        fprintf(rep, "Iteración %02d: %.5f ms\n", i+1, ms);
    }
    fprintf(rep, "----------------------------------------------\n\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// ===================================================
//                      MAIN
// ===================================================
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

    int degreeBins = DEGREE_BINS;
    int rBins = R_BINS;

    float rMax = sqrtf(width*width + height*height) / 2.f;
    float rScale = (float)rBins / (2.f*rMax);

    // Host pre-cálculo
    float *h_cos = (float*)malloc(sizeof(float) * degreeBins);
    float *h_sin = (float*)malloc(sizeof(float) * degreeBins);
    for (int i = 0; i < degreeBins; i++) {
        float rad = i * DEG2RAD;
        h_cos[i] = cosf(rad);
        h_sin[i] = sinf(rad);
    }

    // Device
    unsigned char* d_img;
    float *d_cos_, *d_sin_;
    int* d_acc;
    cudaMalloc(&d_img, sizeof(unsigned char) * imgSize);
    cudaMalloc(&d_cos_, sizeof(float) * degreeBins);
    cudaMalloc(&d_sin_, sizeof(float) * degreeBins);
    cudaMalloc(&d_acc, sizeof(int) * degreeBins * rBins);

    cudaMemcpy(d_img, img->data, sizeof(unsigned char)*imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos_, h_cos, sizeof(float)*degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin_, h_sin, sizeof(float)*degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_cos, h_cos, sizeof(float)*degreeBins);
    cudaMemcpyToSymbol(d_sin, h_sin, sizeof(float)*degreeBins);

    mkdir("reporte", 0777);
    FILE* rep = fopen("reporte/bitacora_tiempos.txt", "a");
    fprintf(rep, "================ BITÁCORA TRANSFORMADA HOUGH ================\n");
    fprintf(rep, "Imagen: %s\nResolución: %dx%d\nIteraciones: %d\n\n",
            argv[1], width, height, ITERACIONES);

    // --------------------
    //   FASE 1 (GLOBAL)
    // --------------------
    ejecutarKernelGlobal(rep, d_img, d_acc, d_cos_, d_sin_,
                         width, height, rBins, degreeBins, rMax, rScale);

    int* h_acc_global = (int*)calloc(degreeBins*rBins, sizeof(int));
    cudaMemcpy(h_acc_global, d_acc, sizeof(int)*degreeBins*rBins, cudaMemcpyDeviceToHost);

    dibujarLineasColor("output_global.ppm", img, h_acc_global,
                       width, height, rBins, degreeBins, rMax, rScale,
                       0,0,255);

    guardarTransformadaHough("hough_global.ppm", h_acc_global, rBins, degreeBins);


    // --------------------
    //   FASE 2 (CONST)
    // --------------------
    ejecutarKernelConst(rep, d_img, d_acc,
                        width, height, rBins, degreeBins, rMax, rScale);

    int* h_acc_const = (int*)calloc(degreeBins*rBins, sizeof(int));
    cudaMemcpy(h_acc_const, d_acc, sizeof(int)*degreeBins*rBins, cudaMemcpyDeviceToHost);

    dibujarLineasColor("output_const.ppm", img, h_acc_const,
                       width, height, rBins, degreeBins, rMax, rScale,
                       0,255,0);

    guardarTransformadaHough("hough_const.ppm", h_acc_const, rBins, degreeBins);


    // --------------------
    //   FASE 3 (SHARED)
    // --------------------
    ejecutarKernelShared(rep, d_img, d_acc,
                         width, height, rBins, degreeBins, rMax, rScale);

    int* h_acc_shared = (int*)calloc(degreeBins*rBins, sizeof(int));
    cudaMemcpy(h_acc_shared, d_acc, sizeof(int)*degreeBins*rBins, cudaMemcpyDeviceToHost);

    dibujarLineasColor("output_shared.ppm", img, h_acc_shared,
                       width, height, rBins, degreeBins, rMax, rScale,
                       255,0,0);

    guardarTransformadaHough("hough_shared.ppm", h_acc_shared, rBins, degreeBins);

    fprintf(rep, "RESULTADOS GUARDADOS.\n\n");
    fclose(rep);

    // Liberación
    cudaFree(d_img);
    cudaFree(d_cos_);
    cudaFree(d_sin_);
    cudaFree(d_acc);
    free(h_cos);
    free(h_sin);
    free(h_acc_global);
    free(h_acc_const);
    free(h_acc_shared);
    freePGM(img);

    return 0;
}
