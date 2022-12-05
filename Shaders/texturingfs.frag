#version 330 core

in vec2 TexCoords;

out vec4 FragColor;

uniform sampler2D sprite;

void main() {

	vec4 color = texture(sprite, TexCoords);

	if(color.a < 0.1) {
		discard;
	}

	FragColor = color;
}