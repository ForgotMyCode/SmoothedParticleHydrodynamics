#include <Image.h>

namespace imageUtil {

	std::vector<unsigned char> pixelsToNormalFormat(Image& image) {
		int const depth = image.spectrum();
		int const nPixels = image.width() * image.height();


		std::vector<unsigned char> pixels;
		pixels.reserve(depth * nPixels);

		std::vector<int> pointers(depth);

		for(int i = 0; i < depth; ++i) {
			pointers[i] = nPixels * i;
		}

		unsigned char* data = image.mirror("y").data();

		for(int i = 0; i < nPixels; ++i) {
			for(auto& p : pointers) {
				pixels.emplace_back(data[p++]);
			}
		}

		return pixels;

	}
}
