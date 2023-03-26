#include <iostream>
#include <vector>
#include <complex>
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <future>

using namespace std;

// Create a new pixel map for the image based on the specified width and height
vector<sf::Uint8> createPixMap(int width, int height);

// Create an array of complex numbers based on the specified width and height
vector<complex<double>> createComplexNums(int width, int height);

// Create a component of the complex numbers array
void createComponent(vector<double>& component, int size);

// Convert the intensities array to a pixels array
void buildPixelsArray(const vector<int>& intensities, vector<sf::Uint8>& pixels);

// Build an array of intensities for a subset of the complex numbers array
void buildIntensitiesArray(size_t arrayStart, size_t arrayEnd, const vector<complex<double>>& complexNums, vector<int>& intensities);

class Func // Class representing the fractal function and its properties (TODO: Make dynamic)
{
    public:
        double realModifier, imaginaryModifier;
        int threshold;

    Func() : realModifier(-0.654), imaginaryModifier(0.475), threshold(2) {}

    // Function to count the number of iterations of the function for a specific coordinate - this is recursive and called until
    // the threshold value is reached or the function has been called 255 times
    int FuncCount(int count, complex<double> coords) 
    {
        if (abs(coords) < threshold) 
        {
            return count < 255 ? FuncCount(count + 1, applyFunc(coords)) : count;
        }

        else 
        {
            return count;
        }
    }

    private:
        // Function to apply the fractal function to a specific coordinate
        complex<double> applyFunc(complex<double> coords) 
        {
            double realNumber = pow(real(coords), 2) - pow(imag(coords), 2) + realModifier;
            double imaginaryNumber = 2 * (imag(coords) * real(coords)) + imaginaryModifier;
            return complex<double>(realNumber, imaginaryNumber);
        }
};

int main() 
{
    int width, height;

    sf::Image image;
    sf::Texture imageTexture;
    sf::Sprite imageSprite;

    vector <sf::Uint8> pixels;

    sf::RenderWindow window;
    sf::View view;

    window.create(sf::VideoMode::getDesktopMode(), "Fractal Fabricator", sf::Style::Fullscreen);

    view = window.getDefaultView();
    window.setView(view);

    width = window.getSize().x;
    height = window.getSize().y;

    imageTexture.create(width, height);
    imageTexture.setSmooth(true);

    pixels = createPixMap(width, height);

    image.create(width, height, pixels.data());

    imageTexture.update(image);

    imageSprite = sf::Sprite(imageTexture);

    imageSprite.setOrigin(imageSprite.getLocalBounds().width / 2, imageSprite.getLocalBounds().height / 2);

    imageSprite.setPosition(((float)width / 2), ((float)height / 2));

    while (window.isOpen()) 
    {
        sf::Event currentEvent;
        window.pollEvent(currentEvent);

        window.clear();

        window.draw(imageSprite);

        window.display();

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) 
        {
            window.close();
        }
    }
}

vector<sf::Uint8> createPixMap(int width, int height) 
{
    size_t size = width * height * 4;

    vector<sf::Uint8> pixels(size);

    Func function;

    vector<complex<double>> complexNums;

    complexNums = createComplexNums(width, height);

    size_t num_threads = std::thread::hardware_concurrency();

    size_t chunk_size = complexNums.size() / num_threads;

    vector<vector<int>> threadIntensities(num_threads);
    vector<std::future<void>> futures;

    for (size_t i = 0; i < num_threads; ++i) 
    {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads - 1) ? complexNums.size() : (i + 1) * chunk_size;

        // Preallocate the required space for each thread's intensities vector
        threadIntensities[i].resize(end - start);

        // Use a lambda function to call buildIntensitiesArray with correct arguments
        futures.push_back(std::async(std::launch::async, [start, end, &complexNums, &threadIntensities, i]()
        {
            buildIntensitiesArray(start, end, std::ref(complexNums), std::ref(threadIntensities[i]));
        }));
    }

    for (auto& future : futures)
    {
        future.wait();
    }

    vector<int> mergedIntensities;

    mergedIntensities.reserve(complexNums.size());

    for (const auto& intensities : threadIntensities)
    {
        mergedIntensities.insert(mergedIntensities.end(), intensities.begin(), intensities.end());
    }

    buildPixelsArray(mergedIntensities, pixels);

    return pixels;
}

void buildPixelsArray(const vector<int>& intensities, vector<sf::Uint8>& pixels) 
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

void buildIntensitiesArray(size_t arrayStart, size_t arrayEnd, const vector<complex<double>>& complexNums, vector<int>& intensities) 
{
    Func function;

    if (arrayEnd > complexNums.size() || arrayStart >= arrayEnd || intensities.size() != arrayEnd - arrayStart) 
    {
        return;
    }

    for (size_t i = arrayStart; i < arrayEnd; ++i) 
    {
        intensities[i - arrayStart] = function.FuncCount(0, complexNums[i]);
    }
}

vector<complex<double>> createComplexNums(int width, int height) 
{
    vector<complex<double>> complexArray(width * height);

    vector<double> real, imaginary;

    createComponent(real, width);
    createComponent(imaginary, height);

    for (int i = 0; i < height; i++) 
    {
        for (int j = 0; j < width; j++) 
        {
            complexArray[i * width + j] = complex<double>(real[j], imaginary[i]);
        }
    }

    return complexArray;
}

void createComponent(vector<double>& component, int size) 
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