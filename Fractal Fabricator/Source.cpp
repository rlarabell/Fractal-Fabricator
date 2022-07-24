#include <iostream>
#include <thread> 
#include <complex> 
#include <SFML/Graphics.hpp> 
#include <SFML/System.hpp>

using namespace std;

vector<sf::Uint8> createPixMap(int width, int height);

vector<complex<double>> createComplexNums(int width, int height);

void createComponent(vector<double>& component, int size);

class Func
{
	public:
		double realMod, imaginMod; 
		int threshold; 

		Func()
		{
			realMod = -0.835;
			imaginMod = 0.2321;
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
		//Function is f(x) = x^2 + realMod + imaginMod
		complex<double> applyFunc(complex<double> coords)
		{
			double realNum, imaginNum;

			//new real number = real^2 - imaginary^2 + realMod (subtracting imaginary squared since i^2 is equal to -1)
			realNum = (real(coords) * real(coords)) - (imag(coords) * imag(coords)) + realMod;

			//new imaginary number = 2 * (real * imaginary) + imaginMod cause FOIL and stuff...
			imaginNum = 2 * (imag(coords) * real(coords)) + imaginMod;

			coords = complex<double>(realNum, imaginNum); 

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

	window.create(sf::VideoMode::getDesktopMode(), "Freaky Fractals", sf::Style::Fullscreen);

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

	imageSprite.setPosition((width / 2), (height / 2));

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

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
		{
			image.saveToFile("C:\\Users\\rlara\\test2.png");
		}
	}
}

vector<sf::Uint8> createPixMap(int width, int height)
{
	int size = width * height * 4;

	vector<sf::Uint8> pixels;

	Func function;

	vector<complex<double>> complexNums;

	vector<int> intensities;

	complexNums = createComplexNums(width, height);

	for (const auto& i : complexNums)
	{
		intensities.push_back(function.FuncCount(0, i));
	}

	for (int i : intensities)
	{
		pixels.push_back(255);   // r 
		pixels.push_back(255);  //  g 
		pixels.push_back(255); //   b 
		pixels.push_back(i);  //    a
	}

	return pixels; 
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

	/*
				  ╔═════════════════════════════════════════╗
				  ║	   MAP REAL AND IMAG VALUES TO PIXELS   ║
				  ║    --------------------------------     ║
				  ║ ►Cycles through each value in intensties║
				  ║  array and sets the corresponding pixel ║
				  ║  to white [(255,255,255) in RGB]        ║
				  ║  with the proper alpha intensity        ║
				  ║	                                        ║
				  ║ ►Loop runs (width*height) times         ║
				  ║                                         ║
				  ║ ►Loop stores (width*height*4) values    ║
				  ╚═════════════════════════════════════════╝
	*/
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
	/*
				  ╔═════════════════════════════════════════╗
				  ║	      CALCULATE ARRAY OF VALUES         ║
				  ║    --------------------------------     ║
				  ║ ►Cycles through each value in intensties║
				  ║  array and sets the corresponding pixel ║
				  ║  to white [(255,255,255) in RGB]        ║
				  ║  with the proper alpha intensity        ║
				  ║	                                        ║
				  ║ ►Loop runs (width*height) times         ║
				  ║                                         ║
				  ║ ►Loop stores (width*height*4) values    ║
				  ╚═════════════════════════════════════════╝
	*/
	for (int i = -1; i < size - 1; i++) // loop runs width number of times starting at -1
	{
		if (component.empty() == true) // if no values have been computed yet
			component.push_back(i); // first real value set to -1

		else 
			component.push_back(component.back() + 2.0 / (size - 1)); // otherwise the next value of the set
																	  // is computed as 2.0/(size-1) more than the
																	 // previous value
	}
}