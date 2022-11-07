#include <Geometry.h>

#include <glad/glad.h>
#include <numeric>

Geometry::Geometry(std::initializer_list<int32> attributeSizes, bool isDataStatic, enum class Geometry::EPrimitive primitive) 
	:
	Primitive(primitive),
	DataUsageMode(isDataStatic ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW)
{
	glGenVertexArrays(1, &this->Handle);
	PARANOID_CHECK();

	glGenBuffers(1, &this->DataHandle);
	PARANOID_CHECK();

	{
		auto guard = Use();

		int32 idx = 0;
		int32 offset = 0;

		this->ElementSize = std::accumulate(attributeSizes.begin(), attributeSizes.end(), int64(0));
		check(ElementSize > 0);

		
		glBindBuffer(GL_ARRAY_BUFFER, this->DataHandle);
		PARANOID_CHECK();

		for(auto const attributeSize : attributeSizes) {
			glVertexAttribPointer(idx, attributeSize, GL_FLOAT, GL_FALSE, this->ElementSize * sizeof(float), reinterpret_cast<void*>(offset * sizeof(float)));
			PARANOID_CHECK();

			glEnableVertexAttribArray(idx);
			PARANOID_CHECK();

			offset += attributeSize;
			++idx;
		}

		glBindBuffer(GL_ARRAY_BUFFER, this->DataHandle);
		PARANOID_CHECK();

	}

	switch(primitive) {
	case EPrimitive::Triangle:
		this->PrimitiveMode = GL_TRIANGLES;
		break;
	case EPrimitive::Quad:
		this->PrimitiveMode = GL_QUADS;
		break;
	case EPrimitive::Line:
		this->PrimitiveMode = GL_LINES;
		break;
	default:
		check(false);
	}
}

void Geometry::FinishChangingGeometry() {
	glBindBuffer(GL_ARRAY_BUFFER, this->DataHandle);
	PARANOID_CHECK();

	glBufferData(GL_ARRAY_BUFFER, this->Buffer.size() * sizeof(float), this->Buffer.data(), this->DataUsageMode);
	PARANOID_CHECK();
	
	this->NumElements = this->Buffer.size() / this->ElementSize;

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	PARANOID_CHECK();
}

void Geometry::Render() {
	glDrawArrays(this->PrimitiveMode, 0, GLsizei(this->NumElements));
}

[[nodiscard]] Guard<Geometry> Geometry::Use() {

	glBindVertexArray(this->Handle);
	PARANOID_CHECK();

	return Guard<Geometry>(this);
}

void Geometry::Unuse() {
	glBindVertexArray(0);
	PARANOID_CHECK();
}