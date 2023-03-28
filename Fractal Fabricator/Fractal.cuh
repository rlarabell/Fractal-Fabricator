#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <SFML/System.hpp>

/// <summary>
/// CUDA kernel function that calculates the intensity of each pixel in the image by applying a function until it has either been called 255 times or it reaches the given
/// threshold value.
/// </summary>
/// <param name="real">An array of real coordinates</param>
/// <param name="imaginary">An array of imaginary coordinates</param>
/// <param name="intensities">An array of intensity values</param>
/// <param name="width">The width of the image</param>
/// <param name="height">The height of the image</param>
/// <param name="realModifier">The real modifier for the function to apply</param>
/// <param name="imaginaryModifier">The imaginary modifier for the function to apply</param>
/// <param name="threshold">The threshold value for the function</param>
/// <returns>Nothing, but the intensities array will be altered as a result of this function</returns>
__global__ void calculateIntensities(const double* real, const double* imaginary, int* intensities, int width, int height, double realModifier, double imaginaryModifier, int threshold);


/// <summary>
/// Create a new pixel map for an image based on the width and height that is given to it
/// </summary>
/// <param name="width">The width of the image</param>
/// <param name="height">The height of the image</param>
/// <returns>An array of Uint8 values - these represent the RGBA values of pixels for each pixel in the image</returns>
std::vector<sf::Uint8> createPixMap(int width, int height);


/// <summary>
/// Create a component of the complex numbers array (i.e. real values based on the desired width or imaginary values based on the desired height)
/// </summary>
/// <param name="component">An array of doubles to place the generated component values into</param>
/// <param name="size">The size that the array needs to be (width of image for real numbers or height of image for imaginary numbers)</param>
/// <returns>Nothing, but the component array will be altered as a result of this function</returns>
void createComponent(std::vector<double>& component, int size);


/// <summary>
/// Convert the intensities array to a pixels array
/// </summary>
/// <param name="intensities">An array containing the alpha value of each pixel (from 0 to 255)</param>
/// <param name="pixels">An array containing the RGBA values of each pixel. I.e. the first four values map to one pixel.</param>
/// <returns>Nothing, but the pixels array will be altered as a result of this function</returns>
void buildPixelsArray(const std::vector<int>& intensities, std::vector<sf::Uint8>& pixels);