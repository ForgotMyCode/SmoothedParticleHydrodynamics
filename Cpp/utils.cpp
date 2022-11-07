#include <utils.h>

#include <random>

namespace utils {

	float random() {
		static std::default_random_engine e;
		static std::uniform_real_distribution<> dis(0.f, 1.f);
		return dis(e);
	}

	
	float random(float low, float high) {
		return random() * (high - low) + low;
	}

}