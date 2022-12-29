/*****************************************************************//**
 * \file   Texture.h
 * \brief  Wrapper around openGL textures.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <initializer_list>
#include <vector>

#include <core.h>
#include <Guard.h>
#include <Image.h>

class Texture {

public:
	/**
	 * \brief Constructor.
	 * 
	 * \param image Original image reference.
	 * \param hasAlphaChannel Whether or not this image uses alpha channel.
	 */
	Texture(Image& image, bool hasAlphaChannel = true);

	/**
	 * \brief Stop this texture from being used. Sgould be called by the guard.
	 * 
	 */
	void Unuse();

	/**
	 * \brief Start using this texture. This texture can be used as long as its guard will be alive.
	 * 
	 * \param textureSlot Texture offset from GL_TEXTURE0 for openGL.
	 * \return Guard.
	 */
	[[nodiscard]] Guard<Texture> Use(uint32 textureSlot = 0);

	/**
	 * \brief Get this texture's current openGL offset from GL_TEXTURE0.
	 * 
	 * \return The offset
	 */
	int32 GetTextureSlot() {
		return this->TextureSlot;
	}

private:
	// openGL handle
	uint32 Handle{};

	// current openGL offset from GL_TEXTURE0
	int32 TextureSlot{ 0 };
};
