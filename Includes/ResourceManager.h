/*****************************************************************//**
 * \file   ResourceManager.h
 * \brief  Manager that holds named resource references.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <vector>

#include <unordered_map>

#include <core.h>

class Geometry;
class Shader;
class Texture;

class ResourceManager {
public:
	using ResourceKeyType = uint64;

	/**
	 * \brief Add new shader to the register.
	 * 
	 * \param name Unique resource name.
	 * \param shader Shader pointer.
	 * \return 
	 */
	ResourceKeyType AddShader(std::string const& name, Shader* shader) {
		ResourceKeyType const key = Shaders.size();

		Shaders.emplace_back(shader);

		check(ResourceKeyMap.find(name) == ResourceKeyMap.end());

		ResourceKeyMap[name] = key;

		return key;
	}

	/**
	 * \brief Add new geometry to the register.
	 * 
	 * \param name Unique resource name.
	 * \param geometry Geometry pointer.
	 * \return 
	 */
	ResourceKeyType AddGeometry(std::string const& name, Geometry* geometry) {
		ResourceKeyType const key = Geometries.size();

		Geometries.emplace_back(geometry);

		check(ResourceKeyMap.find(name) == ResourceKeyMap.end());

		ResourceKeyMap[name] = key;

		return key;
	}

	/**
	 * \brief Add new texture to the register.
	 * 
	 * \param name Unique resource name.
	 * \param texture Texture pointer.
	 * \return 
	 */
	ResourceKeyType AddTexture(std::string const& name, Texture* texture) {
		ResourceKeyType const key = Textures.size();

		Textures.emplace_back(texture);

		check(ResourceKeyMap.find(name) == ResourceKeyMap.end());

		ResourceKeyMap[name] = key;

		return key;
	}

	/**
	 * \brief Get shader using a key.
	 * 
	 * \param key Key to the shader.
	 * \return The shader.
	 */
	Shader* GetShader(ResourceKeyType key) {
		check(key < Shaders.size());

		return Shaders[key];
	}

	/**
	 * \brief Get geometry using a key.
	 * 
	 * \param key Key to the geometry.
	 * \return The geometry.
	 */
	Geometry* GetGeometry(ResourceKeyType key) {
		check(key < Geometries.size());

		return Geometries[key];
	}

	/**
	 * \brief Get texture using a key.
	 * 
	 * \param key Key to the texture.
	 * \return The texture.
	 */
	Texture* GetTexture(ResourceKeyType key) {
		check(key < Textures.size());

		return Textures[key];
	}

	/**
	 * \brief Given resource name, get the key to it.
	 * 
	 * \param name Resource name.
	 * \return The key.
	 */
	ResourceKeyType ResolveNameToKey(std::string const& name) {
		auto result = ResourceKeyMap.find(name);

		check(result != ResourceKeyMap.end());

		return result->second;
	}

private:
	// maps names to keys
	std::unordered_map<std::string, ResourceKeyType> ResourceKeyMap{};

	// maps keys to shaders
	std::vector<Shader*> Shaders{};

	// maps keys to geometries
	std::vector<Geometry*> Geometries{};

	// maps keys to textures
	std::vector<Texture*> Textures{};
};