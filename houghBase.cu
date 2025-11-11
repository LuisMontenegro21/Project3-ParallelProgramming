#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "pgm.h"

// Tablas trigonométricas almacenadas en memoria constante
__constant__ float d_cosTable[180];
__constant__ float d_sinTable[180];

// Kernel que usa memoria global y constante
__global__ void houghKernelGlobalConst(
    const unsigned char* d_image, int width, int height,
    int* d_accumulator, int rhoMax)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixel = d_image[y * width + x];
    if (pixel > 250) {
        for (int t = 0; t < 180; t++) {
            float r = x * d_cosTable[t] + y * d_sinTable[t];
            int rIdx = (int)(r + rhoMax / 2);
            if (rIdx >= 0 && rIdx < rhoMax)
                atomicAdd(&d_accumulator[t * rhoMax + rIdx], 1);
        }
    }
}

// Kernel que usa memoria compartida para mejorar la reutilización de datos
__global__ void houghKernelShared(
    const unsigned char* d_image, int width, int height,
    int* d_accumulator, const float* d_cosTable, const float* d_sinTable,
    int rhoMax)
{
    __shared__ unsigned char sharedPixels[16][16];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    sharedPixels[threadIdx.y][threadIdx.x] = d_image[y * width + x];
    __syncthreads();

    int pixel = sharedPixels[threadIdx.y][threadIdx.x];
    if (pixel > 250) {
        for (int t = 0; t < 180; t++) {
            float r = x * d_cosTable[t] + y * d_sinTable[t];
            int rIdx = (int)(r + rhoMax / 2);
            if (rIdx >= 0 && rIdx < rhoMax)
                atomicAdd(&d_accumulator[t * rhoMax + rIdx], 1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Uso: %s <imagen.pgm>\n", argv[0]);
        return 1;
    }

    // 1. Lectura de la imagen en escala de grises
    PGMImage* img = readPGM(argv[1]);
    int width = img->width;
    int height = img->height;
    int rhoMax = (int)(sqrtf(width * width + height * height));
    printf("Imagen cargada: %dx%d, rhoMax = %d\n", width, height, rhoMax);

    // 2. Generación de las tablas de cosenos y senos
    float* h_cosTable = (float*)malloc(180 * sizeof(float));
    float* h_sinTable = (float*)malloc(180 * sizeof(float));
    for (int t = 0; t < 180; t++) {
        float theta = t * M_PI / 180.0f;
        h_cosTable[t] = cosf(theta);
        h_sinTable[t] = sinf(theta);
    }

    // 3. Reserva de memoria en GPU
    unsigned char* d_image;
    int* d_accumulator;
    float *d_cosTable, *d_sinTable;

    cudaMalloc(&d_image, width * height * sizeof(unsigned char));
    cudaMalloc(&d_accumulator, 180 * rhoMax * sizeof(int));
    cudaMalloc(&d_cosTable, 180 * sizeof(float));
    cudaMalloc(&d_sinTable, 180 * sizeof(float));

    cudaMemcpy(d_image, img->data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cosTable, h_cosTable, 180 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sinTable, h_sinTable, 180 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 4. Ejecución del kernel con memoria global y constante
    cudaMemset(d_accumulator, 0, 180 * rhoMax * sizeof(int));
    cudaMemcpyToSymbol(d_cosTable, h_cosTable, 180 * sizeof(float));
    cudaMemcpyToSymbol(d_sinTable, h_sinTable, 180 * sizeof(float));

    printf("Ejecutando kernel con memoria Global + Constante...\n");
    houghKernelGlobalConst<<<grid, block>>>(d_image, width, height, d_accumulator, rhoMax);
    cudaDeviceSynchronize();

    int* h_accumulator = (int*)malloc(180 * rhoMax * sizeof(int));
    cudaMemcpy(h_accumulator, d_accumulator, 180 * rhoMax * sizeof(int), cudaMemcpyDeviceToHost);
    writePGM("hough_output_const.pgm", h_accumulator, rhoMax, 180);
    printf("Resultado guardado en hough_output_const.pgm\n");

    // 5. Ejecución del kernel con memoria compartida
    cudaMemset(d_accumulator, 0, 180 * rhoMax * sizeof(int));
    printf("Ejecutando kernel con memoria Compartida...\n");
    houghKernelShared<<<grid, block>>>(d_image, width, height, d_accumulator,
                                       d_cosTable, d_sinTable, rhoMax);
    cudaDeviceSynchronize();

    cudaMemcpy(h_accumulator, d_accumulator, 180 * rhoMax * sizeof(int), cudaMemcpyDeviceToHost);
    writePGM("hough_output_shared.pgm", h_accumulator, rhoMax, 180);
    printf("Resultado guardado en hough_output_shared.pgm\n");

    // 6. Liberación de recursos
    cudaFree(d_image);
    cudaFree(d_accumulator);
    cudaFree(d_cosTable);
    cudaFree(d_sinTable);
    free(h_cosTable);
    free(h_sinTable);
    free(h_accumulator);
    freePGM(img);

    printf("Ejecución completada correctamente.\n");
    return 0;
}
