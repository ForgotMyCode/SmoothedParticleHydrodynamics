#version 330 core

in vec2 TexCoords;

out vec4 FragColor;

uniform sampler2D sprite;

uniform float density;

uniform int gridX;
uniform int gridY;
uniform int gridZ;

void main() {
	float d = clamp(density, 0.0, 1.0);
	float b = clamp(d, 0.0, 0.1);

	FragColor = vec4(gridX, gridY, gridZ, d);
	//FragColor = vec4(d, 1.0 - d, (0.1 - b) * 10.0, 1.0);
}