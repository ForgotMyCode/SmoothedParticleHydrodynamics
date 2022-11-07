#include <Loader.h>

#include <fstream>
#include <sstream>

#include <core.h>

namespace loader {
	std::string loadTextFromFile(std::string fileName) {
		std::ifstream fstream(fileName);
		check(!fstream.fail());
		std::stringstream ss;
		ss << fstream.rdbuf();
		check(!fstream.fail());
		return ss.str();
	}

	Image loadImage(std::string fileName) {
		Image image(fileName.c_str());
		return image;
	}
}