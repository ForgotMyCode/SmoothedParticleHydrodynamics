/*****************************************************************//**
 * \file   Mesh.h
 * \brief  Mesh - geometry in scene that has a shader attached.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

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

	/**
	 * \brief Constructor.
	 * 
	 * \param geometry Pointer to its geometry.
	 * \param shader Pointer to its shader.
	 */
	Mesh(Geometry* geometry, Shader* shader)
		: GeometryPtr(geometry), ShaderPtr(shader)
	{
		RecalculateModelMatrix();
		UpdateHandles();
	}

	/**
	 * \brief Change geometry of this mesh.
	 * 
	 * \param geometry New geometry pointer.
	 */
	void SetGeometry(Geometry* geometry) {
		this->GeometryPtr = geometry;
	}

	/**
	 * \brief Change shader of this mesh.
	 * 
	 * \param shader New shader of this mesh.
	 */
	void SetShader(Shader* shader) {
		this->ShaderPtr = shader;
		UpdateHandles();
	}

	/**
	 * \brief Render this geometry.
	 * 
	 */
	virtual void Render() override;

	/**
	 * \brief Get world location.
	 * 
	 * \return vec3 location.
	 */
	auto GetLocation() const {
		return this->Location;
	}

	/**
	 * \brief Set world location.
	 * 
	 * \param location New location.
	 */
	void SetLocation(glm::vec3 const& location) {
		this->Location = location;
		RecalculateModelMatrix();
	}

	/**
	 * \brief Get world scale (uniform).
	 * 
	 * \return Scale.
	 */
	float GetScale() const {
		return this->Scale;
	}

	/**
	 * \brief Set new uniform scale.
	 * 
	 * \param scale New scale.
	 */
	void SetScale(float scale) {
		this->Scale = scale;
		RecalculateModelMatrix();
	}

	/**
	 * \brief Recalculate the model matrix.
	 * 
	 */
	void RecalculateModelMatrix() {
		glm::mat4 modelMatrix(1.0f);
		modelMatrix = glm::scale(modelMatrix, glm::vec3(this->Scale));
		modelMatrix = glm::translate(modelMatrix, this->Location);

		this->ModelMatrix = modelMatrix;
	}

	/**
	 * \brief Get the cached model matrix.
	 * 
	 * \return mat4 model matrix.
	 */
	auto const& GetCachedModelMatrix() const {
		return this->ModelMatrix;
	}

	/**
	 * \brief Update handles to fit current shader.
	 * 
	 */
	virtual void UpdateHandles();

protected:
	// pointer to geometry
	Geometry* GeometryPtr{};

	// pointer to shader
	Shader* ShaderPtr{};

	// model matrix
	glm::mat4 ModelMatrix{};

	// world location
	glm::vec3 Location{};
	
	// uniform scale
	float Scale{1.f};

	// shader handles
	int32 ModelHandle{}, ViewHandle{}, ProjectionHandle{};
};

class TexturedMesh : public Mesh {
	SETSUPER(Mesh);
	
public:
	/**
	 * \brief Constructor.
	 * 
	 * \param geometry Geometry.
	 * \param shader Shader.
	 * \param texture Texture.
	 */
	TexturedMesh(Geometry* geometry, Shader* shader, Texture* texture)
		:
		Super(geometry, shader),
		TexturePtr(texture)
	{
		TexturedMesh::UpdateHandles();
	}

	/**
	 * \brief Set new texture.
	 * 
	 * \param texture New texture pointer.
	 */
	void SetTexture(Texture* texture) {
		this->TexturePtr = texture;
	}

	/**
	 * \brief Render this textured mesh.
	 * 
	 */
	virtual void Render() override;

	/**
	 * \brief Update handles to fit current shader.
	 * 
	 */
	virtual void UpdateHandles() override;

protected:

	// texture
	Texture* TexturePtr{};

	// texture shader handle
	int32 TextureHandle{};
};