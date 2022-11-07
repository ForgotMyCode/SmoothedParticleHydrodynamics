#pragma once

#include <initializer_list>
#include <vector>

#include <core.h>
#include <Guard.h>
#include <Image.h>

class Texture {

public:

	Texture(Image& image, bool hasAlphaChannel = true);

	void Unuse();

	[[nodiscard]] Guard<Texture> Use(uint32 textureSlot = 0);

	int32 GetTextureSlot() {
		return this->TextureSlot;
	}

private:

	uint32 Handle{};

	int32 TextureSlot{ 0 };
};