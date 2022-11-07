#pragma once

#include <libpng16/png.h>
#define cimg_use_png 1
#include <CImg.h>
#include <vector>

using Image = cimg_library::CImg<unsigned char>;

namespace imageUtil {
	
	std::vector<unsigned char> pixelsToNormalFormat(Image& image);

}