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

	ResourceKeyType AddShader(std::string const& name, Shader* shader) {
		ResourceKeyType const key = Shaders.size();

		Shaders.emplace_back(shader);

		check(ResourceKeyMap.find(name) == ResourceKeyMap.end());

		ResourceKeyMap[name] = key;

		return key;
	}

	ResourceKeyType AddGeometry(std::string const& name, Geometry* geometry) {
		ResourceKeyType const key = Geometries.size();

		Geometries.emplace_back(geometry);

		check(ResourceKeyMap.find(name) == ResourceKeyMap.end());

		ResourceKeyMap[name] = key;

		return key;
	}

	ResourceKeyType AddTexture(std::string const& name, Texture* texture) {
		ResourceKeyType const key = Textures.size();

		Textures.emplace_back(texture);

		check(ResourceKeyMap.find(name) == ResourceKeyMap.end());

		ResourceKeyMap[name] = key;

		return key;
	}

	Shader* GetShader(ResourceKeyType key) {
		check(key < Shaders.size());

		return Shaders[key];
	}

	Geometry* GetGeometry(ResourceKeyType key) {
		check(key < Geometries.size());

		return Geometries[key];
	}

	Texture* GetTexture(ResourceKeyType key) {
		check(key < Textures.size());

		return Textures[key];
	}

	ResourceKeyType ResolveNameToKey(std::string const& name) {
		auto result = ResourceKeyMap.find(name);

		check(result != ResourceKeyMap.end());

		return result->second;
	}

private:
	std::unordered_map<std::string, ResourceKeyType> ResourceKeyMap{};
	std::vector<Shader*> Shaders{};
	std::vector<Geometry*> Geometries{};
	std::vector<Texture*> Textures{};
};