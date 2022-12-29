/*****************************************************************//**
 * \file   Geometry.h
 * \brief  3D Geometry model.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <initializer_list>
#include <vector>

#include <core.h>
#include <Guard.h>


class Geometry {
public:
	enum class EPrimitive : uint8 {
		Triangle,
		Quad,
		Line
	};

	/**
	 * \brief Constructor.
	 * 
	 * \param attributeSizes Defines the format. For every element, define its attribute lengths.
	 * \param isDataStatic Whether or not are the data going to change.
	 * \param primitive Primitives. Triangles by default.
	 */
	Geometry(std::initializer_list<int32> attributeSizes, bool isDataStatic = true, EPrimitive primitive = EPrimitive::Triangle);
	
	/**
	 * \brief Get direct access to buffer for changing.
	 * 
	 * \return Geometry buffer reference.
	 */
	std::vector<float>& BeginChangingGeometry() {
		return this->Buffer;
	}

	/**
	 * \brief Declare you have finished making changes after calling BeginChangingGeometry. The changes will be propagated.
	 * 
	 */
	void FinishChangingGeometry();

	/**
	 * \brief Draw this geometry.
	 * 
	 */
	void Render();


	void Unuse();

	/**
	 * \brief Bind this geometry for usage.
	 * 
	 * \return Guard. Its destruction will stop this geometry from being used.
	 */
	[[nodiscard]] Guard<Geometry> Use();

private:
	// internal geometry buffer. May be modified.
	std::vector<float> Buffer{};

	// Primitive
	EPrimitive Primitive{};

	// Vertex array handle
	uint32 Handle{};

	// Buffer handle
	uint32 DataHandle{};

	// Primitive in OpenGL perspective
	uint32 PrimitiveMode{};

	// Is data static in OpenGL perspective
	uint32 DataUsageMode{};

	// Size of element
	uint64 ElementSize{};

	// Number of elements
	uint64 NumElements{};
};