#include <utils.h>

#include <random>

namespace utils {

	float random() {
		static thread_local std::default_random_engine e;
		static thread_local std::uniform_real_distribution<> dis(0.f, 1.f);
		return dis(e);
	}

	
	float random(float low, float high) {
		return random() * (high - low) + low;
	}

}