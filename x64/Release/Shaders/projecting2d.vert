#version 330 core
layout (location = 0) in vec2 pos;
layout (location = 1) in vec2 tex;

out vec2 TexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
	TexCoords = tex;

	vec4 origin = model * vec4(0.0, 0.0, 0.0, 1.0);
	vec4 originViewSpace = view * origin;

	originViewSpace.xy += pos;

	gl_Position = projection * originViewSpace;
}