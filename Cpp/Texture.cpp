#include <Texture.h>

#include <glad/glad.h>

Texture::Texture(Image& image, bool hasAlphaChannel) {

	glGenTextures(1, &this->Handle);
	PARANOID_CHECK();

	auto guard = Use();

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	PARANOID_CHECK();

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	PARANOID_CHECK();

	auto pixels = imageUtil::pixelsToNormalFormat(image);

	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		hasAlphaChannel ? GL_RGBA : GL_RGB,
		image.width(), image.height(),
		0,
		hasAlphaChannel ? GL_RGBA : GL_RGB,
		GL_UNSIGNED_BYTE,
		pixels.data()
	);
	PARANOID_CHECK();

	glGenerateMipmap(GL_TEXTURE_2D);
	PARANOID_CHECK();
}

void Texture::Unuse() {
	glBindTexture(GL_TEXTURE_2D, 0);
	this->TextureSlot = 0;
}

[[nodiscard]] Guard<Texture> Texture::Use(uint32 textureSlot) {
	this->TextureSlot = textureSlot;
	glActiveTexture(GL_TEXTURE0 + textureSlot);
	glBindTexture(GL_TEXTURE_2D, this->Handle);
	PARANOID_CHECK();

	return Guard<Texture>(this);
}