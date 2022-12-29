/*****************************************************************//**
 * \file   ParticleSimulatingMesh.h
 * \brief  Textured mesh that holds and manages the particle simulation.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <Mesh.h>
#include <Simulation/Simulation.h>

class ParticleSimulatingMesh : public TexturedMesh {
	SETSUPER(TexturedMesh);

public:
	/**
	 * \brief Constructor.
	 * 
	 * \param geometry Geometry of the particle.
	 * \param shader Shader of the particle.
	 * \param texture Texture of the particle.
	 */
	ParticleSimulatingMesh(Geometry* geometry, Shader* shader, Texture* texture) : Super(geometry, shader, texture) {}

	/**
	 * \brief Called when the scene is initialized.
	 *
	 */
	virtual void Begin() override;

	/**
	 * \brief Called every frame.
	 * 
	 * \param deltaSeconds Time since the last frame in seconds.
	 */
	virtual void Step(float deltaSeconds) override;

	/**
	 * \brief Render the particle system.
	 * 
	 */
	virtual void Render() override;

private:
	// Particle simulation that does the actual computation
	Simulation ParticleSimulation{};
};