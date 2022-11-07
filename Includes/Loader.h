#pragma once

#include <string>

#include <Image.h>

namespace loader {


	std::string loadTextFromFile(std::string fileName);

	Image loadImage(std::string fileName);

}