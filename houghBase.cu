#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "common/pgm.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Parámetros de la transformada de Hough
#define DEGREE_INC 2               // Incremento de ángulo (2 grados)
#define DEGREE_BINS (180 / DEGREE_INC)
#define R_BINS 100                 // Número de divisiones de r

// Kernel CUDA: calcula el acumulador de la transformada de Hough
__global__ void houghTransformKernel(
    int* d_image, int width, int height,
    int* d_acc, int rBins, int degreeBins, float rScale, float rMax)
{
    int gloX = blockIdx.x * blockDim.x + threadIdx.x;
    int gloY = blockIdx.y * blockDim.y + threadIdx.y;

    if (gloX >= width || gloY >= height) return;

    int idx = gloY * width + gloX;
    int pixel = d_image[idx];
    if (pixel < 255) return;

    float xCoord = gloX - (width / 2.0f);
    float yCoord = (height / 2.0f) - gloY;

    for (int t = 0; t < degreeBins; t++) {
        float theta = t * DEGREE_INC * M_PI / 180.0f;
        float r = xCoord * cosf(theta) + yCoord * sinf(theta);
        int rIdx = (int)((r + rMax) * rScale);
        if (rIdx >= 0 && rIdx < rBins) {
            atomicAdd(&d_acc[t * rBins + rIdx], 1);
        }
    }
}

// Función host: transformada de Hough
void houghTransformCPU(const char* inputFile, const char* outputFile) {
    // Leer imagen
    PGMImage* img = readPGM(inputFile);
    int width = img->width;
    int height = img->height;
    int imgSize = width * height;

    printf("Imagen cargada: %d x %d\n", width, height);

    // Reservar memoria en el host para el acumulador
    int degreeBins = DEGREE_BINS;
    int rBins = R_BINS;
    float rMax = sqrtf(width * width + height * height);
    float rScale = (float)rBins / (2.0f * rMax);
    int accSize = degreeBins * rBins * sizeof(int);
    int* h_acc = (int*)calloc(degreeBins * rBins, sizeof(int));

    // Reservar memoria en el device
    int *d_image, *d_acc;
    cudaMalloc((void**)&d_image, imgSize * sizeof(int));
    cudaMalloc((void**)&d_acc, accSize);

    // Inicializar acumulador en device
    cudaMemset(d_acc, 0, accSize);

    // Copiar imagen al device
    cudaMemcpy(d_image, img->data, imgSize * sizeof(int), cudaMemcpyHostToDevice);

    // Configuración del grid y bloques
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Llamada al kernel
    houghTransformKernel<<<gridDim, blockDim>>>(d_image, width, height,
        d_acc, rBins, degreeBins, rScale, rMax);

    cudaDeviceSynchronize();

    // Copiar acumulador de vuelta al host
    cudaMemcpy(h_acc, d_acc, accSize, cudaMemcpyDeviceToHost);

    // Generar imagen de salida visualizando el acumulador
    int maxVal = 0;
    for (int i = 0; i < degreeBins * rBins; i++)
        if (h_acc[i] > maxVal) maxVal = h_acc[i];

    int* accImage = (int*)malloc(degreeBins * rBins * sizeof(int));
    for (int i = 0; i < degreeBins * rBins; i++)
        accImage[i] = (int)(255.0f * ((float)h_acc[i] / maxVal));

    writePGM(outputFile, accImage, rBins, degreeBins);

    printf("Transformada completada. Imagen guardada como '%s'\n", outputFile);

    // Liberar memoria
    free(h_acc);
    free(accImage);
    free(img->data);
    free(img);
    cudaFree(d_image);
    cudaFree(d_acc);
}

int main() {
    const char* inputFile = "runway.pgm";
    const char* outputFile = "hough_output.pgm";
    houghTransformCPU(inputFile, outputFile);
    return 0;
}
