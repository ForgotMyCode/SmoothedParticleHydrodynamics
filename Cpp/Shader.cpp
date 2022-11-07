#include <Shader.h>

#include <iostream>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

CompiledShaderUnit::~CompiledShaderUnit() noexcept {
	glDeleteShader(this->Handle);
	PARANOID_CHECK();
}

Shader::Shader(std::initializer_list<CompiledShaderUnit> units) {
	this->Handle = glCreateProgram();
	PARANOID_CHECK();

	for(auto const& unit : units) {
		glAttachShader(this->Handle, unit.GetHandle());
		PARANOID_CHECK();
	}

	glLinkProgram(this->Handle);
	PARANOID_CHECK();

	int ok = 0;

	glGetProgramiv(this->Handle, GL_LINK_STATUS, &ok);
	PARANOID_CHECK();

	int logLength = 0;
	glGetProgramiv(this->Handle, GL_INFO_LOG_LENGTH, &logLength);
	PARANOID_CHECK();

	if(!ok || logLength > 1) {

		char* infoLog = new char[logLength]();
		glGetProgramInfoLog(this->Handle, logLength, nullptr, infoLog);

		std::cerr << "Shader link error: (" << logLength << " chars):" << std::string(infoLog) << "\n";

		delete[] infoLog;

		check(false);
	}
}

Guard<Shader> Shader::Use() {
	glUseProgram(this->Handle);
	PARANOID_CHECK();
	return Guard<Shader>(this);
}

void Shader::Unuse() {
	glUseProgram(0);
}

int32 Shader::GetUniformHandle(char const* name) {
	GLint const uniformHandle = glGetUniformLocation(this->Handle, name);
	PARANOID_CHECK();
	return uniformHandle;
}

void Shader::PassUniformMatrix4f(int32 uniformHandle, glm::mat4 const& matrix) {
	glUniformMatrix4fv(uniformHandle, 1, GL_FALSE, glm::value_ptr(matrix));
	PARANOID_CHECK();
}

void Shader::PassUniformInt(int32 uniformHandle, int32 integer) {
	glUniform1i(uniformHandle, integer);
	PARANOID_CHECK();
}

namespace shaderCompiler {

	CompiledShaderUnit compileShaderFromSource(std::string const& source, ShaderType shaderType) {

		GLenum mappedShaderType{};

		switch(shaderType) {
			case ShaderType::VertexShader:
				mappedShaderType = GL_VERTEX_SHADER;
				break;
			case ShaderType::FragmentShader:
				mappedShaderType = GL_FRAGMENT_SHADER;
				break;
			default:
				check(false);
		}

		auto cSource = source.c_str();

		uint32 handle = glCreateShader(mappedShaderType);
		PARANOID_CHECK();

		glShaderSource(handle, 1, &cSource, nullptr);
		PARANOID_CHECK();

		glCompileShader(handle);
		PARANOID_CHECK();

		int ok = 0;
		glGetShaderiv(handle, GL_COMPILE_STATUS, &ok);
		PARANOID_CHECK();

		if(!ok) {
			int logLength = 0;
			glGetProgramiv(handle, GL_INFO_LOG_LENGTH, &logLength);
			PARANOID_CHECK();

			char* infoLog = new char[logLength]();

			glGetShaderInfoLog(handle, logLength, nullptr, infoLog);
			PARANOID_CHECK();
			
			std::cerr << "Shader compile error: " << std::string(infoLog) << "\n";

			delete[] infoLog;

			check(false);
		}

		return CompiledShaderUnit(handle);
	}

}