#include <config.h>

#include <charconv>
#include <fstream>
#include <iostream>
#include <string>

namespace config {

	bool isGPUsimulation = true;

	int32 stepsPerFrame = 10;

	namespace simulation {

		//float particleSize = 0.01f;

		int64 maxNumberOfParticles = 5000;

		float gravityDirection = 0.f;

		float gravityDirectionChangePerSecond = 1.f;
	
		namespace physics {

			//float gasConstantK = 0.05f;
			float gasConstantK = 20.f; //0.05f;

			//float restDensity = 100.f;
			float restDensity = 1000.f;//1000.f;

			float outOfBoundsVelocityScale = 1.f;//0.9f;

			float gravityConstant = 9.77f;

			//float particleMass = 0.1f;
			float particleMass = 0.02f;

			//float dynamicViscosity = 0.07f;
			float dynamicViscosity = 0.07f;//1.f;

			float timeScale = 1.0f;

		}

	}

	void loadConfigFromFile(std::string const& fileName) {
		std::ifstream input(fileName);
		
		if(!input.is_open()) {
			std::cerr << "Could not open config file, default configuration will be used!\n";
			return;
		}

		auto parseFloat = [](std::string_view raw, float& value) -> bool {
			if(std::from_chars(raw.data(), raw.data() + raw.length(), value).ec != std::errc{}) {
				std::cerr << "Could not parse value " << raw << ", skipping...\n";
				return false;
			}
			return true;
		};

		auto parseInt = [](std::string_view raw, int32& value) -> bool {
			if(std::from_chars(raw.data(), raw.data() + raw.length(), value).ec != std::errc{}) {
				std::cerr << "Could not parse value " << raw << ", skipping...\n";
				return false;
			}
			return true;
		};

		std::string line;

		while(std::getline(input, line)) {
			size_t const equalsPosition = line.find('=');

			if(equalsPosition == std::string::npos) {
				std::cerr << "Could not parse line " << line << ", skipping..." << '\n';
				continue;
			}

			std::string_view keyName(line.begin(), std::next(line.begin(), equalsPosition));
			std::string_view valueName(std::next(line.begin(), equalsPosition + 1), line.end());

			if(keyName == "isGPUsimulation") {
				int32 value;
				if(!parseInt(valueName, value)) {
					continue;
				}

				isGPUsimulation = value != 0;
			}
			else if(keyName == "stepsPerFrame") {
				int32 value;
				if(!parseInt(valueName, value)) {
					continue;
				}

				stepsPerFrame = value;
			}
			else if(keyName == "gravityDirectionChangePerSecond") {
				float value;
				if(!parseFloat(valueName, value)) {
					continue;
				}

				simulation::gravityDirectionChangePerSecond = value;
			}
			else if(keyName == "gasConstantK") {
				float value;
				if(!parseFloat(valueName, value)) {
					continue;
				}

				simulation::physics::gasConstantK = value;
			}
			else if(keyName == "restDensity") {
				float value;
				if(!parseFloat(valueName, value)) {
					continue;
				}

				simulation::physics::restDensity = value;
			}
			else if(keyName == "outOfBoundsVelocityScale") {
				float value;
				if(!parseFloat(valueName, value)) {
					continue;
				}

				simulation::physics::outOfBoundsVelocityScale = value;
			}
			else if(keyName == "gravityConstant") {
				float value;
				if(!parseFloat(valueName, value)) {
					continue;
				}

				simulation::physics::gravityConstant = value;
			}
			else if(keyName == "particleMass") {
				float value;
				if(!parseFloat(valueName, value)) {
					continue;
				}

				simulation::physics::particleMass = value;
			}
			else if(keyName == "dynamicViscosity") {
				float value;
				if(!parseFloat(valueName, value)) {
					continue;
				}

				simulation::physics::dynamicViscosity = value;
			}
			else if(keyName == "timeScale") {
				float value;
				if(!parseFloat(valueName, value)) {
					continue;
				}

				simulation::physics::timeScale = value;
			}
			else if(keyName == "maxNumberOfParticles") {
				int32 value;
				if(!parseInt(valueName, value)) {
					continue;
				}

				simulation::maxNumberOfParticles = value;
			}

		}

		input.close();
	}

}