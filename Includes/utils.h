/*****************************************************************//**
 * \file   utils.h
 * \brief  Various math extensions/utility functions.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <random>

namespace utils {
	/**
	 * \brief Compute integer power of a float possibly at compile time. The power is assumed to be very low.
	 * 
	 * \param x Base number.
	 * \param p Power.
	 * \return Result of x ** p.
	 */
	constexpr float powf(float x, int p) {
		float result = 1.f;

		for(int i = 0; i < p; ++i) {
			result *= x;
		}

		return result;
	}

	/**
	 * \brief Square a float possibly at compile time.
	 * 
	 * \param x The float.
	 * \return Result of x * x.
	 */
	constexpr float square(float x) {
		return x * x;
	}

	/**
	 * \brief Cube a float possibly at compile time.
	 * 
	 * \param x The float.
	 * \return Result of x * x * x.
	 */
	constexpr float cube(float x) {
		return x * x * x;
	}

	/**
	 * \brief Determines whether or not a float is sufficiently close to 0.
	 *
	 * \param x Number.
	 * \param tolerance Max distance from 0 that is still considered 0.
	 * \return true if x can be considered 0.
	 */
	constexpr bool isCloseToZero(float x, float tolerance = 0.000001f) {
		return x < tolerance && x > -tolerance;
	}

	/**
	 * \brief Generates a random uniform number between 0 and 1.
	 *	The generation is thread_local so it is thread safe.
	 * 
	 * \return Random number.
	 */
	float random();

	/**
	 * \brief Generates a random uniform number in range.
	 *	The generation is thread_local so it is thread safe.
	 * 
	 * \param low Low end of the range.
	 * \param high High end of the range.
	 * \return Random number between low and high.
	 */
	float random(float low, float high);

}