#pragma once

#include <Mesh.h>
#include <Simulation/Simulation.h>

class ParticleSimulatingMesh : public TexturedMesh {
	SETSUPER(TexturedMesh);

public:
	ParticleSimulatingMesh(Geometry* geometry, Shader* shader, Texture* texture) : Super(geometry, shader, texture) {}

	virtual void Begin() override;

	virtual void Step(float deltaSeconds) override;

	virtual void Render() override;

private:
	Simulation ParticleSimulation{};
};