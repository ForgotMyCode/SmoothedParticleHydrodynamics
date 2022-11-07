#pragma once

#include <array>
#include <vector>
#include <iterator>

namespace shapes {

	template<float minX, float maxX, float minY, float maxY, float minZ, float maxZ>
	constexpr auto axisAlignedBox() {
		return std::array<float, 6 * 6 * 3>{
			{
				minX, minY, minZ,
				maxX, minY, minZ,
				maxX, maxY, minZ,
				maxX, maxY, minZ,
				minX, maxY, minZ,
				minX, minY, minZ,

				minX, minY, maxZ,
				maxX, minY, maxZ,
				maxX, maxY, maxZ,
				maxX, maxY, maxZ,
				minX, maxY, maxZ,
				minX, minY, maxZ,

				minX, maxY, maxZ,
				minX, maxY, minZ,
				minX, minY, minZ,
				minX, minY, minZ,
				minX, minY, maxZ,
				minX, maxY, maxZ,

				maxX, maxY, maxZ,
				maxX, maxY, minZ,
				maxX, minY, minZ,
				maxX, minY, minZ,
				maxX, minY, maxZ,
				maxX, maxY, maxZ,

				minX, minY, minZ,
				maxX, minY, minZ,
				maxX, minY, maxZ,
				maxX, minY, maxZ,
				minX, minY, maxZ,
				minX, minY, minZ,

				minX, maxY, minZ,
				maxX, maxY, minZ,
				maxX, maxY, maxZ,
				maxX, maxY, maxZ,
				minX, maxY, maxZ,
				minX, maxY, minZ,
			}
		};
	}

	template<float minX, float maxX, float minY, float maxY, float minZ, float maxZ>
	constexpr auto axisAlignedLinedBox() {
		return std::array<float, 12*2*3 + 6*4*3>{
			{
				minX, minY, minZ, // xX yy
				maxX, minY, minZ,

				maxX, minY, minZ, // XX yY
				maxX, maxY, minZ,

				maxX, maxY, minZ, // xX YY
				minX, maxY, minZ,

				minX, maxY, minZ, // xx yY
				minX, minY, minZ,

				minX, minY, maxZ, // xX yy
				maxX, minY, maxZ,

				maxX, minY, maxZ, // XX yY
				maxX, maxY, maxZ,

				maxX, maxY, maxZ, // xX YY
				minX, maxY, maxZ,

				minX, maxY, maxZ, // xx yY
				minX, minY, maxZ,

				minX, minY, minZ,
				minX, minY, maxZ,

				minX, maxY, minZ,
				minX, maxY, maxZ,				

				maxX, minY, minZ,
				maxX, minY, maxZ,

				maxX, maxY, minZ,
				maxX, maxY, maxZ,

				minX, minY, minZ, // xX yY
				maxX, maxY, minZ,
				minX, maxY, minZ, // xX Yy
				maxX, minY, minZ,

				minX, minY, maxZ, // xX yY
				maxX, maxY, maxZ,
				minX, maxY, maxZ, // xX Yy
				maxX, minY, maxZ,

				minX, minY, minZ,
				maxX, minY, maxZ,
				minX, minY, maxZ,
				maxX, minY, minZ,

				minX, maxY, minZ,
				maxX, maxY, maxZ,
				minX, maxY, maxZ,
				maxX, maxY, minZ,

				minX, minY, minZ,
				minX, maxY, maxZ,
				minX, minY, maxZ,
				minX, maxY, minZ,
				   		 
				maxX, minY, minZ,
				maxX, maxY, maxZ,
				maxX, minY, maxZ,
				maxX, maxY, minZ,
			}
		};
	}

	template<float minX, float maxX, float minY, float maxY>
	constexpr auto texturedUniquad() {
		return std::array<float, 3 * 2 * 4> { {
			minX, minY, 0.0f, 0.0f,
			minX, maxY, 0.0f, 1.0f,
			maxX, minY, 1.0f, 0.0f,

			maxX, minY, 1.0f, 0.0f,
			minX, maxY, 0.0f, 1.0f,
			maxX, maxY, 1.0f, 1.0f,

		}};
	}
}