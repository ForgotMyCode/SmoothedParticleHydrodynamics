#include <defines.h>

#include <glad/glad.h>
#include <iostream>
#include <string>

#include <core.h>

void checkOpenGLerror(char const* file, int line) {
	GLenum const errorCode = glGetError();

	[[unlikely]]
	if (errorCode != GL_NO_ERROR) {
		std::cerr << "OpenGL error: " << errorCode << "\n\tAt " << std::string(file) << ":" << line << "\n";
		check(false);
	}
}