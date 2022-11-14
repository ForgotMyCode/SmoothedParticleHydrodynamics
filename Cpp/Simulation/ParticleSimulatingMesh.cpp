#include <Simulation/ParticleSimulatingMesh.h>

#include <glad/glad.h>

#include <Geometry.h>
#include <Shader.h>
#include <Texture.h>
#include <Window.h>

void ParticleSimulatingMesh::Begin() {
	this->ParticleSimulation.Initialize();
}

void ParticleSimulatingMesh::Step(float deltaSeconds) {
	this->ParticleSimulation.cpuStepParallel(deltaSeconds);
}

void ParticleSimulatingMesh::Render() {
	
		auto geometryGuard = this->GeometryPtr->Use();
		PARANOID_CHECK();

		auto shaderGuard = this->ShaderPtr->Use();
		PARANOID_CHECK();

		auto textureGuard = this->TexturePtr->Use();

		this->ShaderPtr->PassUniformInt(this->TextureHandle, this->TexturePtr->GetTextureSlot());

		Window* window = Window::GetActiveWindow();
		
		this->ShaderPtr->PassUniformMatrix4f(this->ViewHandle, window->GetCamera().GetCachedViewMatrix());
		PARANOID_CHECK();
		this->ShaderPtr->PassUniformMatrix4f(this->ProjectionHandle, window->GetCachedProjectionMatrix());
		PARANOID_CHECK();

		const int32 densityHandle = this->ShaderPtr->GetUniformHandle("density");

		auto modelMatrix = this->GetCachedModelMatrix();

		const int32 gridXhandle = this->ShaderPtr->GetUniformHandle("gridX");
		const int32 gridYhandle = this->ShaderPtr->GetUniformHandle("gridY");
		const int32 gridZhandle = this->ShaderPtr->GetUniformHandle("gridZ");


		for(int32 i = 0; i < this->ParticleSimulation.GetNumberOfParticles(); ++i) {
			Simulation::Particle& particle = ParticleSimulation.GetParticles()[i];

			auto particleModel = glm::translate(modelMatrix, particle.Position);

			this->ShaderPtr->PassUniformMatrix4f(this->ModelHandle, particleModel);
			PARANOID_CHECK();

			this->ShaderPtr->PassUniformFloat(densityHandle, particle.Density);
			PARANOID_CHECK();

			this->ShaderPtr->PassUniformInt(gridXhandle, particle.CellIdx[0] % 2);
			this->ShaderPtr->PassUniformInt(gridYhandle, particle.CellIdx[1] % 2);
			this->ShaderPtr->PassUniformInt(gridZhandle, particle.CellIdx[2] % 2);

			this->GeometryPtr->Render();
			PARANOID_CHECK();
		}
}
