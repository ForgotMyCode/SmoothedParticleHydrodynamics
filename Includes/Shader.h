#pragma once

#include <initializer_list>
#include <string>
#include <glm/glm.hpp>

#include <core.h>
#include <Guard.h>

enum class ShaderType : uint8 {
	VertexShader,
	FragmentShader
};

class CompiledShaderUnit {
public:
	CompiledShaderUnit(uint32 handle) : Handle(handle) {}

	~CompiledShaderUnit() noexcept;

	uint32 GetHandle() const {
		return Handle;
	}

private:
	uint32 Handle{};
};

class Shader {
public:
	Shader() = default;

	Shader(std::initializer_list<CompiledShaderUnit> units);

	void Unuse();

	[[nodiscard]] Guard<Shader> Use();

	auto GetHandle() const{
		return this->Handle;
	}

	int32 GetUniformHandle(char const* name);

	void PassUniformMatrix4f(int32 uniformHandle, glm::mat4 const& matrix);

	void PassUniformInt(int32 uniformHandle, int32 integer);

private:
	uint32 Handle{};
};

namespace shaderCompiler {
	
	CompiledShaderUnit compileShaderFromSource(std::string const& source, ShaderType shaderType);

}