#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <SceneObject.h>
#include <core.h>

class Geometry;
class Shader;
class Texture;

class Mesh : public SceneObject {

public:
	Mesh(Geometry* geometry, Shader* shader)
		: GeometryPtr(geometry), ShaderPtr(shader)
	{
		RecalculateModelMatrix();
		UpdateHandles();
	}

	void SetGeometry(Geometry* geometry) {
		this->GeometryPtr = geometry;
	}

	void SetShader(Shader* shader) {
		this->ShaderPtr = shader;
		UpdateHandles();
	}

	virtual void Render() override;

	auto GetLocation() const {
		return this->Location;
	}

	void SetLocation(glm::vec3 const& location) {
		this->Location = location;
		RecalculateModelMatrix();
	}

	float GetScale() const {
		return this->Scale;
	}

	void SetScale(float scale) {
		this->Scale = scale;
		RecalculateModelMatrix();
	}

	void RecalculateModelMatrix() {
		glm::mat4 modelMatrix(1.0f);
		modelMatrix = glm::translate(modelMatrix, this->Location);
		modelMatrix = glm::scale(modelMatrix, glm::vec3(this->Scale));

		this->ModelMatrix = modelMatrix;
	}

	auto const& GetCachedModelMatrix() const {
		return this->ModelMatrix;
	}

	virtual void UpdateHandles();

protected:
	Geometry* GeometryPtr{};
	Shader* ShaderPtr{};

	glm::mat4 ModelMatrix{};

	glm::vec3 Location{};
	float Scale{1.f};

	int32 ModelHandle{}, ViewHandle{}, ProjectionHandle{};
};

class TexturedMesh : public Mesh {
	SETSUPER(Mesh);
	
public:

	TexturedMesh(Geometry* geometry, Shader* shader, Texture* texture)
		:
		Super(geometry, shader),
		TexturePtr(texture)
	{
		TexturedMesh::UpdateHandles();
	}

	void SetTexture(Texture* texture) {
		this->TexturePtr = texture;
	}

	virtual void Render() override;

	virtual void UpdateHandles() override;

protected:

	Texture* TexturePtr{};

	int32 TextureHandle{};
};