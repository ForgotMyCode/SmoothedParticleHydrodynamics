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

	Geometry(std::initializer_list<int32> attributeSizes, bool isDataStatic = true, EPrimitive primitive = EPrimitive::Triangle);
	
	std::vector<float>& BeginChangingGeometry() {
		return this->Buffer;
	}

	void FinishChangingGeometry();

	void Render();

	void Unuse();

	[[nodiscard]] Guard<Geometry> Use();

private:
	std::vector<float> Buffer{};

	EPrimitive Primitive{};

	uint32 Handle{};
	uint32 DataHandle{};
	uint32 PrimitiveMode{};
	uint32 DataUsageMode{};

	uint64 ElementSize{};
	uint64 NumElements{};
};