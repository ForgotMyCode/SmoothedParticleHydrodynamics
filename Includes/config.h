#pragma once

#include <glm/glm.hpp>
#include <numbers>

#include <defines.h>
#include <niceIntTypes.h>
#include <utils.h>
#include <cudaCompatibleConfig.h>

namespace config {

	CPPGLOBAL bool isGPUsimulation;

	CPPGLOBAL int32 stepsPerFrame;
	
	namespace debug {

		CONSTANT bool noChecks = false;

	}

	namespace window {
		
		CONSTANT int64 defaultWidth = 1280;

		CONSTANT int64 defaultHeight = 720;

		CONSTANT char const* defaultTitle = "mezlondr: Smoothed Particle Hydrodynamics";

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
	
		CPPGLOBAL int64 maxNumberOfParticles;

		CPPGLOBAL float particleSize;

		CONSTANT float minUpdateDelaySecs = 0.1f;

		CPPGLOBAL float gravityDirection;

		CPPGLOBAL float gravityDirectionChangePerSecond;

		namespace physics {

			//CONSTANT float smoothingLength = 8.f;
			//CONSTANT float smoothingLength = 0.15f;

			CONSTANT float smoothingLengthSquared = smoothingLength * smoothingLength;

			CPPGLOBAL float particleMass;

			CPPGLOBAL float gasConstantK;

			CPPGLOBAL float restDensity;

			CPPGLOBAL float outOfBoundsVelocityScale;

			CPPGLOBAL float gravityConstant;

			CPPGLOBAL float dynamicViscosity;

			CPPGLOBAL float timeScale;

			inline
			namespace magicConstants {

				CONSTANT float smoothingKernelNormalizationDistanceToDensityConstant = (315.f) / (64.f * std::numbers::pi_v<float> * utils::powf(smoothingLength, 9));

				CONSTANT float smoothingKernelNormalizationPressureToForceConstant = (-45.f) / (std::numbers::pi_v<float> * utils::powf(smoothingLength, 6));

				CONSTANT float smoothingKernelNormalizationViscousForceConstant = (45.f) / (std::numbers::pi_v<float> * utils::powf(smoothingLength, 6));
			
			}

		}

		namespace spawning {
			
			CONSTANT bool isSpawningNewParticlesEnabled = true;

			CONSTANT float spawnDelaySecs = 1.f;

		}

		namespace boundingBox {
		//	
		//	//CONSTANT float minX = -20.f;
		//	CONSTANT float minX = -2.f;

		//	//CONSTANT float maxX = 20.f;
		//	CONSTANT float maxX = 2.f;

		//	CONSTANT float minY = 0.f;

		//	CONSTANT float maxY = 4.f;

		//	//CONSTANT float minZ = -20.f;
		//	CONSTANT float minZ = -2.f;

		//	//CONSTANT float maxZ = 20.f;
		//	CONSTANT float maxZ = 2.f;

		//	CONSTANT float xSize = maxX - minX;

		//	CONSTANT float ySize = maxY - minY;

		//	CONSTANT float zSize = maxZ - minZ;

			CONSTANT glm::vec3 mins(config::simulation::boundingBox::minX, config::simulation::boundingBox::minY, config::simulation::boundingBox::minZ);

			CONSTANT glm::vec3 maxs(config::simulation::boundingBox::maxX, config::simulation::boundingBox::maxY, config::simulation::boundingBox::maxZ);

			CONSTANT glm::vec3 sizes(config::simulation::boundingBox::xSize, config::simulation::boundingBox::ySize, config::simulation::boundingBox::zSize);

		inline
		namespace grid {
		//		// using cheap ceiling at compile time, for some reason cmath does not use constexpr *yet*, it will mostly work fine

		//		CONSTANT int32 xSamples = int32((xSize + 0.99f * physics::smoothingLength) / physics::smoothingLength);

		//		CONSTANT int32 ySamples = int32((ySize + 0.99f * physics::smoothingLength) / physics::smoothingLength);

		//		CONSTANT int32 zSamples = int32((zSize + 0.99f * physics::smoothingLength) / physics::smoothingLength);

				CONSTANT glm::vec<3, int32> samples(config::simulation::boundingBox::xSamples, config::simulation::boundingBox::ySamples, config::simulation::boundingBox::zSamples);

		//		CONSTANT int32 nCells = xSamples * ySamples * zSamples;

			}
		}
	
	}

	/**
	 * \brief Load config from file.
	 * 
	 * \param fileName Name or path of the config file.
	 */
	void loadConfigFromFile(std::string const& fileName);

}