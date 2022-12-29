/*****************************************************************//**
 * \file   Shader.h
 * \brief  Wrapper around OpenGL shaders.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

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
	/**
	 * \brief Constructor.
	 * 
	 * \param handle Shader handle
	 */
	CompiledShaderUnit(uint32 handle) : Handle(handle) {}

	/**
	 * \brief Destructor. Deletes this shader.
	 * 
	 * \return 
	 */
	~CompiledShaderUnit() noexcept;

	/**
	 * \brief Get handle to this shader.
	 * 
	 * \return 
	 */
	uint32 GetHandle() const {
		return Handle;
	}

private:
	// Shader in perspective of OpenGL.
	uint32 Handle{};
};

class Shader {
public:
	/**
	 * \brief Constructor.
	 * 
	 */
	Shader() = default;

	/**
	 * \brief Links CompiledShaderUnit s to create a program.
	 * 
	 * \param units List of shader units to link.
	 */
	Shader(std::initializer_list<CompiledShaderUnit> units);

	/**
	 * \brief Deactivate this shader from being used. Should be called by the guard.
	 * 
	 */
	void Unuse();

	/**
	 * \brief Start using this shader. The shader will be used as long as the returned guard is alive.
	 * 
	 * \return 
	 */
	[[nodiscard]] Guard<Shader> Use();

	/**
	 * \brief Get OpenGL handle to this shader.
	 * 
	 * \return The handle.
	 */
	auto GetHandle() const{
		return this->Handle;
	}

	/**
	 * \brief Get OpenGL handle to a name uniform in this shader.
	 * 
	 * \param name Name of the uniform
	 * \return OpenGL handle.
	 */
	int32 GetUniformHandle(char const* name);

	/**
	 * \brief Pass uniform mat4 to this shader given its handle.
	 * 
	 * \param uniformHandle The handle.
	 * \param matrix The matrix.
	 */
	void PassUniformMatrix4f(int32 uniformHandle, glm::mat4 const& matrix);

	/**
	 * \brief Pass uniform int to this shader given its handle.
	 * 
	 * \param uniformHandle The handle.
	 * \param integer The int.
	 */
	void PassUniformInt(int32 uniformHandle, int32 integer);

	/**
	 * \brief Pass uniform float to this shader given its handle.
	 * 
	 * \param uniformHandle The handle.
	 * \param floatingPoint The float.
	 */
	void PassUniformFloat(int32 uniformHandle, float floatingPoint);

private:
	// openGL handle
	uint32 Handle{};
};

namespace shaderCompiler {
	
	/**
	 * \brief Compile shader unit from source code.
	 * 
	 * \param source string source code.
	 * \param shaderType enum shader type.
	 * \return Compiled shader unit.
	 */
	CompiledShaderUnit compileShaderFromSource(std::string const& source, ShaderType shaderType);

}