#include <iostream>
#include <thread> 
#include <complex> 
#include <SFML/Graphics.hpp> 
#include <SFML/System.hpp>

using namespace std;

vector<sf::Uint8> createPixMap(int width, int height);

vector<complex<double>> createComplexNums(int width, int height);

void createComponent(vector<double>& component, int size);

void buildPixelsArray(vector<int> intensities, vector<sf::Uint8>& pixels);

void buildIntensitiesArray(int arrayStart, int arrayEnd, vector<complex<double>> complexNums, vector<int>& intensities);

class Func
{
	public:
		double realModifier, imaginaryModifier; 
		int threshold; 

		Func()
		{
			realModifier = -0.835;
			imaginaryModifier = 0.2321;
			threshold = 2;
		}

		int FuncCount(int count, complex<double> coords)
		{
			if (abs(coords) < threshold) 
			{
				if (count < 255) 
					return FuncCount(count + 1, applyFunc(coords)); 

				else 
					return count;
			}

			else 
				return count; 
		}

	private:
		//Function is f(x) = x^2 + realModifier + imaginaryModifier
		complex<double> applyFunc(complex<double> coords)
		{
			double realNumber, imaginaryNumber;

			//new real number = real^2 - imaginary^2 + realModifier (subtracting imaginary squared since i^2 is equal to -1)
			realNumber = (real(coords) * real(coords)) - (imag(coords) * imag(coords)) + realModifier;

			//new imaginary number = 2 * (real * imaginary) + imaginaryModifier cause FOIL and stuff...
			imaginaryNumber = 2 * (imag(coords) * real(coords)) + imaginaryModifier;

			coords = complex<double>(realNumber, imaginaryNumber); 

			return coords; 
		}
};

int main()
{
	int width, height;

	sf::Image image;
	sf::Texture imageTexture;
	sf::Sprite imageSprite; 

	vector<sf::Uint8> pixels; 

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
	int size = width * height * 4;

	vector<sf::Uint8> pixels;

	Func function;

	vector<complex<double>> complexNums;

	vector<int> intensitiesFirstFourth, intensitiesSecondFourth, intensitiesThirdFourth, intensitiesFourthFourth;

	complexNums = createComplexNums(width, height);

	int complexNumbersQuartered = complexNums.size()/4;

	thread threadFirstFourth, threadSecondFourth, threadThirdFourth, threadFourthFourth;

	threadFirstFourth = thread(buildIntensitiesArray, 0, complexNumbersQuartered, complexNums, ref(intensitiesFirstFourth));
	threadSecondFourth = thread(buildIntensitiesArray, complexNumbersQuartered+1, complexNumbersQuartered*2, complexNums, ref(intensitiesSecondFourth));
	threadThirdFourth = thread(buildIntensitiesArray, complexNumbersQuartered*2+1, complexNumbersQuartered*3, complexNums, ref(intensitiesThirdFourth));
	threadFourthFourth = thread(buildIntensitiesArray, complexNumbersQuartered*3+1, complexNums.size()-1, complexNums, ref(intensitiesFourthFourth));
	
	threadFirstFourth.join();
	threadSecondFourth.join();
	threadThirdFourth.join();
	threadFourthFourth.join();

	buildPixelsArray(intensitiesFirstFourth, pixels);
	buildPixelsArray(intensitiesSecondFourth, pixels);
	buildPixelsArray(intensitiesThirdFourth, pixels);
	buildPixelsArray(intensitiesFourthFourth, pixels);

	return pixels; 
}

void buildPixelsArray(vector<int> intensities, vector<sf::Uint8>& pixels)
{
	for (const auto& i : intensities)
	{
		pixels.push_back(255);   // r 
		pixels.push_back(255);  //  g 
		pixels.push_back(255); //   b 
		pixels.push_back(i);  //    a
	}
}

void buildIntensitiesArray(int arrayStart, int arrayEnd, vector<complex<double>> complexNums, vector<int>& intensities)
{
	Func function;

	for (int i = arrayStart; i < arrayEnd; i++)
	{
		intensities.push_back(function.FuncCount(0, complexNums[i]));
	}
}

vector<complex<double>> createComplexNums(int width, int height)
{
	complex<double> complexNumber; 

	vector<complex<double>> complexArray; 

	vector<double> real, imaginary;

	thread threadReal, threadImaginary;

	threadReal = thread(createComponent, ref(real), width);
	threadImaginary = thread(createComponent, ref(imaginary), height);

	threadReal.join();
	threadImaginary.join();

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			complexNumber = complex<double>(real[j], imaginary[i]);

			complexArray.push_back(complexNumber);
		}
	}

	return complexArray;
}

void createComponent(vector<double>& component, int size)
{
	for (int i = -1; i < size - 1; i++) 
	{
		if (component.empty() == true) // if no values have been computed yet
			component.push_back(i); // first real value set to -1

		else 
			component.push_back(component.back() + 2.0 / (size - 1)); // otherwise the next value of the set
																	  // is computed as 2.0/(size-1) more than the
																	 // previous value
	}
}