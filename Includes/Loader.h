/*****************************************************************//**
 * \file   Loader.h
 * \brief  Resource loading.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <string>

#include <Image.h>

namespace loader {

	/**
	 * \brief Read text (everything) from a text file.
	 * 
	 * \param fileName Name (or path) of the file to read.
	 * \return Contents of the file.
	 */
	std::string loadTextFromFile(std::string fileName);

	/**
	 * \brief Loads an image using CImg.
	 * 
	 * \param fileName Name (or path) of the image to read.
	 * \return The image.
	 */
	Image loadImage(std::string fileName);

}