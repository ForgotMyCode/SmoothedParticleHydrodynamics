#include <config.h>

namespace config {

	bool isGPUsimulation = false;

	int32 stepsPerFrame = 10;

	namespace simulation {

		float particleSize = 0.01f;

		float gravityDirection = 0.f;

		float gravityDirectionChangePerSecond = 1.f;
	
		namespace physics {

			//float gasConstantK = 0.05f;
			float gasConstantK = 0.05f;

			//float restDensity = 100.f;
			float restDensity = 1000.f;

			float outOfBoundsVelocityScale = 0.9f;

			float gravityConstant = 9.77f;

			//float particleMass = 0.1f;
			float particleMass = 0.02f;

			//float dynamicViscosity = 0.07f;
			float dynamicViscosity = 1.f;

			float timeScale = 1.0f;

		}

	}

}