#pragma once

#include <glm/glm.hpp>
#include <numbers>

#include <defines.h>
#include <niceIntTypes.h>
#include <utils.h>

namespace config {
	
	namespace debug {

		CONSTANT bool noChecks = false;

	}

	namespace window {
		
		CONSTANT int64 defaultWidth = 1280;

		CONSTANT int64 defaultHeight = 720;

		CONSTANT char const* defaultTitle = "TODO";

		inline
		namespace camera {

			CONSTANT float nearPlaneDistance = 0.1f;

			CONSTANT float farPlaneDistance = 1000.0f;

			CONSTANT float fovRadians = 0.7853981633974f;

			CONSTANT float movementSpeed = 20.f; // in (opengl units) per second

			CONSTANT float mouseSpeed = 0.002f; // scale

			CONSTANT bool hideCursor = true;

			CONSTANT float maxPitch = 1.41372f;

			CONSTANT float minPitch = -maxPitch;

			CONSTANT float defaultYaw = -1.5707963f;

		}

	}

	namespace render {
		
		namespace clearColor {

			CONSTANT float r = 0.1f;

			CONSTANT float g = 0.1f;

			CONSTANT float b = 0.2f;

		}

	}

	namespace simulation {
	
		CONSTANT int32 maxNumberOfParticles = 50;

		CONSTANT float particleSize = 1.f;

		CONSTANT float minUpdateDelaySecs = 0.1f;

		namespace physics {

			CONSTANT float smoothingLength = 5.f;

			CONSTANT float particleMass = 1.0f;

			CONSTANT float gasConstantK = 20.f;

			CONSTANT float restDensity = 1.0f;

			CONSTANT float outOfBoundsVelocityScale = -0.9f;

			CONSTANT float gravityForce = -0.1f;

			CONSTANT float dynamicViscosity = 0.5f;

			inline
			namespace magicConstants {

				CONSTANT float smoothingKernelNormalizationDistanceToDensityConstant = (315.f * particleMass) / (64.f * std::numbers::pi_v<float> * utils::powf(smoothingLength, 9));

				CONSTANT float smoothingKernelNormalizationPressureToForceConstant = (-45.f * particleMass) / (std::numbers::pi_v<float> * utils::powf(smoothingLength, 6));

				CONSTANT float smoothingKernelNormalizationViscousForceConstant = (45.f * dynamicViscosity) / (std::numbers::pi_v<float> *utils::powf(smoothingLength, 6));
			
			}

		}

		namespace spawning {
			
			CONSTANT bool isSpawningNewParticlesEnabled = true;

			CONSTANT float spawnDelaySecs = 1.f;

		}

		namespace boundingBox {
			
			CONSTANT float minX = -20.f;

			CONSTANT float maxX = 20.f;

			CONSTANT float minY = 0.f;

			CONSTANT float maxY = 75.f;

			CONSTANT float minZ = -20.f;

			CONSTANT float maxZ = 20.f;

			CONSTANT float xSize = maxX - minX;

			CONSTANT float ySize = maxY - minY;

			CONSTANT float zSize = maxZ - minZ;

			CONSTANT glm::vec3 mins(minX, minY, minZ);

			CONSTANT glm::vec3 maxs(maxX, maxY, maxZ);

			CONSTANT glm::vec3 sizes(xSize, ySize, zSize);

			inline
			namespace grid {
				// using cheap ceiling at compile time, for some reason cmath does not use constexpr *yet*, it will mostly work fine

				CONSTANT int32 xSamples = int32((xSize + 0.99f * physics::smoothingLength) / physics::smoothingLength);

				CONSTANT int32 ySamples = int32((ySize + 0.99f * physics::smoothingLength) / physics::smoothingLength);

				CONSTANT int32 zSamples = int32((zSize + 0.99f * physics::smoothingLength) / physics::smoothingLength);

				CONSTANT glm::vec<3, int32> samples(xSamples, ySamples, zSamples);

				CONSTANT int32 nCells = xSamples * ySamples * zSamples;

			}
		}
	
	}

}