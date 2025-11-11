#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "common/pgm.h"
#include <vector>
#include <algorithm>

__constant__ float c_cos[DEGREE_BINS];
__constant__ float c_sin[DEGREE_BINS];

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Parámetros de la transformada de Hough
#define DEGREE_INC 2               // Incremento de ángulo (2 grados)
#define DEGREE_BINS (180 / DEGREE_INC)
#define R_BINS 100                 // Número de divisiones de r

// Kernel CUDA: calcula el acumulador de la transformada de Hough
__global__ void houghTransformKernel(
    const unsigned char* d_image, int width, int height,
    int* d_acc, int rBins, int degreeBins, float rScale, float rMax)
{
    int gloX = blockIdx.x * blockDim.x + threadIdx.x;
    int gloY = blockIdx.y * blockDim.y + threadIdx.y;

    if (gloX >= width || gloY >= height) return;

    int idx = gloY * width + gloX;
    unsigned char pixel = d_image[idx];
    if (pixel < 255) return; // treat only edge (white) pixels

    float xCoord = gloX - (width / 2.0f);
    float yCoord = (height / 2.0f) - gloY;

    for (int t = 0; t < degreeBins; t++) {
        float r = xCoord * c_cos[t] + yCoord * c_sin[t];
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
    cudaMemcpy(d_image, img->data, imgSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Configuración del grid y bloques
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Medición de tiempo con CUDA events
    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    cudaEventRecord(evStart, 0);

    houghTransformKernel<<<gridDim, blockDim>>>(d_image, width, height,
        d_acc, rBins, degreeBins, rScale, rMax);
    cudaEventRecord(evStop, 0);
    cudaEventSynchronize(evStop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, evStart, evStop);
    printf("Tiempo kernel (global acc + const trig): %.3f ms\n", ms);
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);

    cudaMemcpy(h_acc, d_acc, accSize, cudaMemcpyDeviceToHost);

    int maxVal = 0;
    for (int i = 0; i < degreeBins * rBins; i++)
        if (h_acc[i] > maxVal) maxVal = h_acc[i];
    if (maxVal == 0) maxVal = 1;

    int* accImage = (int*)malloc(degreeBins * rBins * sizeof(int));
    for (int i = 0; i < degreeBins * rBins; i++)
        accImage[i] = (int)(255.0f * ((float)h_acc[i] / maxVal));

    writePGM(outputFile, accImage, rBins, degreeBins);

    int voteThreshold = (int)(0.7f * maxVal);
    std::vector<std::pair<int,int>> peaks;
    for (int t = 0; t < degreeBins; ++t) {
        for (int rIdx = 0; rIdx < rBins; ++rIdx) {
            int v = h_acc[t * rBins + rIdx];
            if (v >= voteThreshold) {
                peaks.emplace_back(t, rIdx);
            }
        }
    }

    std::vector<unsigned char> rgb(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        unsigned char g = img->data[i];
        rgb[3*i] = g; rgb[3*i+1] = g; rgb[3*i+2] = g;
    }

    float rScaleInv = (2.0f * rMax) / (float)rBins;

    auto drawPixel = [&](int x, int y, unsigned char rC, unsigned char gC, unsigned char bC) {
        if (x >=0 && x < width && y >=0 && y < height) {
            int idx = y * width + x;
            rgb[3*idx] = rC; rgb[3*idx+1] = gC; rgb[3*idx+2] = bC;
        }
    };

    for (auto &pk : peaks) {
        int t = pk.first; int rIdx = pk.second;
        float theta = t * DEGREE_INC * M_PI / 180.0f;
        float r = rIdx * rScaleInv - rMax;
        float cosT = cosf(theta); float sinT = sinf(theta);

        for (int x = 0; x < width; ++x) {
            float xC = x - width/2.0f;
            float yC;
            if (fabsf(sinT) > 1e-5f) {
                yC = (r - xC * cosT) / sinT;
                int y = (int)(height/2.0f - yC + 0.5f);
                drawPixel(x, y, 255, 0, 0);
            }
        }
    }

    writePPM("hough_lines.ppm", rgb.data(), width, height);

    printf("Transformada completada. Imagen guardada como '%s'\n", outputFile);

    // Liberar memoria
    free(h_acc);
    free(accImage);
    freePGM(img);
    cudaFree(d_image);
    cudaFree(d_acc);
}

int main() {
    const char* inputFile = "runway.pgm";
    const char* outputFile = "hough_output.pgm";
    std::vector<float> hostCos(DEGREE_BINS), hostSin(DEGREE_BINS);
    for (int t = 0; t < DEGREE_BINS; ++t) {
        float theta = t * DEGREE_INC * M_PI / 180.0f;
        hostCos[t] = cosf(theta);
        hostSin[t] = sinf(theta);
    }
    cudaMemcpyToSymbol(c_cos, hostCos.data(), sizeof(float)*DEGREE_BINS);
    cudaMemcpyToSymbol(c_sin, hostSin.data(), sizeof(float)*DEGREE_BINS);
    houghTransformCPU(inputFile, outputFile);
    return 0;
}
