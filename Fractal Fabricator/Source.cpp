#include <vector>
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include "Fractal.cuh"

int main() 
{
    int width, height;

    sf::Image image;
    sf::Texture imageTexture;
    sf::Sprite imageSprite;

    std::vector <sf::Uint8> pixels;

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

