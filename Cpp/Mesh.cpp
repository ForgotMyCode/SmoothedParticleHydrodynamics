#include <Mesh.h>	

#include <glad/glad.h>

#include <Geometry.h>
#include <Shader.h>
#include <Texture.h>
#include <Window.h>

void Mesh::Render() {
	{
		auto geometryGuard = this->GeometryPtr->Use();
		PARANOID_CHECK();

		auto shaderGuard = this->ShaderPtr->Use();
		PARANOID_CHECK();

		Window* window = Window::GetActiveWindow();

		this->ShaderPtr->PassUniformMatrix4f(this->ModelHandle, this->GetCachedModelMatrix());
		PARANOID_CHECK();
		this->ShaderPtr->PassUniformMatrix4f(this->ViewHandle, window->GetCamera().GetCachedViewMatrix());
		PARANOID_CHECK();
		this->ShaderPtr->PassUniformMatrix4f(this->ProjectionHandle, window->GetCachedProjectionMatrix());
		PARANOID_CHECK();

		this->GeometryPtr->Render();
		PARANOID_CHECK();
	}
}

void Mesh::UpdateHandles() {
	ModelHandle = ShaderPtr->GetUniformHandle("model");
	ViewHandle = ShaderPtr->GetUniformHandle("view");
	ProjectionHandle = ShaderPtr->GetUniformHandle("projection");
}

void TexturedMesh::Render() {
	auto shaderGuard = this->ShaderPtr->Use();

	auto textureGuard = this->TexturePtr->Use();

	this->ShaderPtr->PassUniformInt(this->TextureHandle, this->TexturePtr->GetTextureSlot());

	Super::Render();
}

void TexturedMesh::UpdateHandles() {
	Super::UpdateHandles();

	this->TextureHandle = ShaderPtr->GetUniformHandle("sprite");
}