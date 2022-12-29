/*****************************************************************//**
 * \file   Image.h
 * \brief  Utility functions to work with images
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <libpng16/png.h>
#define cimg_use_png 1
#include <CImg.h>
#include <vector>

using Image = cimg_library::CImg<unsigned char>;

namespace imageUtil {
	
	/**
	 * \brief Function that extracts bytes in normal buffer from CImg's structure.
	 */
	std::vector<unsigned char> pixelsToNormalFormat(Image& image);

}