#pragma once

#include <random>

namespace utils {

	constexpr float powf(float x, int p) {
		float result = 1.f;

		for(int i = 0; i < p; ++i) {
			result *= x;
		}

		return result;
	}

	constexpr float square(float x) {
		return x * x;
	}

	constexpr float cube(float x) {
		return x * x * x;
	}

	constexpr bool isCloseToZero(float x, float tolerance = 0.000001f) {
		return x < tolerance && x > -tolerance;
	}

	float random();

	float random(float low, float high);

}