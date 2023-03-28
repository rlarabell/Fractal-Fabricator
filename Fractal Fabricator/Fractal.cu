#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Fractal.cuh"

__global__ void calculateIntensities(const double* real, const double* imaginary, int* intensities, int width, int height, double realModifier, double imaginaryModifier, int threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = y * width + x;

    double realCoord = real[x];
    double imagCoord = imaginary[y];

    double zReal = realCoord;
    double zImag = imagCoord;

    int count = 0;
    while (count < 255 && (zReal * zReal + zImag * zImag) < threshold * threshold)
    {
        double zRealTemp = zReal * zReal - zImag * zImag + realModifier;
        zImag = 2 * zReal * zImag + imaginaryModifier;
        zReal = zRealTemp;
        count++;
    }

    intensities[index] = count;
}

class Func // Class representing the fractal function and its properties (TODO: Make dynamic - also probably make this a struct too now that it's only data)
{
    public:
        double realModifier, imaginaryModifier;
        int threshold;

        Func() : realModifier(-0.654), imaginaryModifier(0.475), threshold(2) {}
};

std::vector<sf::Uint8> createPixMap(int width, int height)
{
    size_t size = width * height * 4;

    std::vector<sf::Uint8> pixels(size);

    Func function;

    std::vector<int> intensities(width * height);

    std::vector<double> real, imaginary;

    createComponent(real, width);
    createComponent(imaginary, height);

    double* d_real, * d_imaginary;
    int* d_intensities;

    cudaMalloc((void**)&d_real, width * sizeof(double));
    cudaMalloc((void**)&d_imaginary, height * sizeof(double));
    cudaMalloc((void**)&d_intensities, width * height * sizeof(int));

    cudaMemcpy(d_real, real.data(), width * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imaginary, imaginary.data(), height * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    calculateIntensities<<<gridDim, blockDim>>>(d_real, d_imaginary, d_intensities, width, height, function.realModifier, function.imaginaryModifier, function.threshold);

    cudaMemcpy(intensities.data(), d_intensities, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_real);
    cudaFree(d_imaginary);
    cudaFree(d_intensities);

    buildPixelsArray(intensities, pixels);

    return pixels;
}

void buildPixelsArray(const std::vector<int>& intensities, std::vector<sf::Uint8>& pixels)
{
    for (size_t i = 0; i < intensities.size(); ++i)
    {
        size_t baseIndex = i * 4;
        pixels[baseIndex] = 255;     // r
        pixels[baseIndex + 1] = 255; // g
        pixels[baseIndex + 2] = 255; // b
        pixels[baseIndex + 3] = intensities[i]; // a
    }
}

void createComponent(std::vector<double>& component, int size)
{
    for (int i = -1; i < size - 1; i++)
    {
        if (component.empty()) // if no values have been computed yet
        {
            component.push_back(i); // first real value set to -1
        }
        else
        {
            component.push_back(component.back() + 2.0 / (size - 1)); // otherwise the next value of the set
                                                                      // is computed as 2.0/(size-1) more than the
                                                                      // previous value
        }
    }
}