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
			CONSTANT float minX = -2.f;

			//CONSTANT float maxX = 20.f;
			CONSTANT float maxX = 2.f;

			CONSTANT float minY = 0.f;

			CONSTANT float maxY = 4.f;

			//CONSTANT float minZ = -20.f;
			CONSTANT float minZ = -2.f;

			//CONSTANT float maxZ = 20.f;
			CONSTANT float maxZ = 2.f;

			CONSTANT float xSize = maxX - minX;

			CONSTANT float ySize = maxY - minY;

			CONSTANT float zSize = maxZ - minZ;

			inline
				namespace grid {
				// using cheap ceiling at compile time, for some reason cmath does not use constexpr *yet*, it will mostly work fine

				CONSTANT int32 xSamples = int32((xSize + 0.99f * physics::smoothingLength) / physics::smoothingLength);

				CONSTANT int32 ySamples = int32((ySize + 0.99f * physics::smoothingLength) / physics::smoothingLength);

				CONSTANT int32 zSamples = int32((zSize + 0.99f * physics::smoothingLength) / physics::smoothingLength);

				//CONSTANT glm::vec<3, int32> samples(xSamples, ySamples, zSamples);

				CONSTANT int32 nCells = xSamples * ySamples * zSamples;

			}
		}
	}
}