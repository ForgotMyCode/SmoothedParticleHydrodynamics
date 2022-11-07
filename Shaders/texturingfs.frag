#version 330 core

in vec2 TexCoords;

out vec4 FragColor;

uniform sampler2D sprite;

void main() {
	vec4 color = texture(sprite, TexCoords);

	if(color.w > 0) {
		FragColor = vec4(color.xyz, 1.0);
	}
	else {
		FragColor = vec4(0.0);
	}
}