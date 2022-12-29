/*****************************************************************//**
 * \file   cudaCompatibleConfig.h
 * \brief  Part of the config the CUDA compiler can handle.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <glm/glm.hpp>
#include <niceIntTypes.h>

#include <defines.h>

namespace config {

	namespace simulation {

		namespace physics {
			
			CONSTANT float smoothingLength = 0.15f;

		}

		namespace boundingBox {

			//CONSTANT float minX = -20.f;

			// bounding box min x
			CONSTANT float minX = -2.f;

			//CONSTANT float maxX = 20.f;

			// bounding box max x
			CONSTANT float maxX = 2.f;

			// bounding box min y
			CONSTANT float minY = 0.f;

			// bounding box max y
			CONSTANT float maxY = 4.f;

			//CONSTANT float minZ = -20.f;

			// bounding box min z
			CONSTANT float minZ = -2.f;

			//CONSTANT float maxZ = 20.f;

			// bounding box max z
			CONSTANT float maxZ = 2.f;

			// size of bounding box's x side
			CONSTANT float xSize = maxX - minX;

			// size of bounding box's y side
			CONSTANT float ySize = maxY - minY;

			// size of bounding box's z side
			CONSTANT float zSize = maxZ - minZ;

			inline
				namespace grid {
				// using cheap ceiling at compile time, for some reason cmath does not use constexpr *yet*, it will mostly work fine

				// cells on x axis of the bounding box's grid
				CONSTANT int32 xSamples = int32((xSize + 0.99f * physics::smoothingLength) / physics::smoothingLength);

				// cells on y axis of the bounding box's grid
				CONSTANT int32 ySamples = int32((ySize + 0.99f * physics::smoothingLength) / physics::smoothingLength);

				// cells on z axis of the bounding box's grid
				CONSTANT int32 zSamples = int32((zSize + 0.99f * physics::smoothingLength) / physics::smoothingLength);

				//CONSTANT glm::vec<3, int32> samples(xSamples, ySamples, zSamples);

				// number of cells of the bounding box's grid
				CONSTANT int32 nCells = xSamples * ySamples * zSamples;

			}
		}
	}
}