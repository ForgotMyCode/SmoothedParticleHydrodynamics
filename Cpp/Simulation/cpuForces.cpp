#include <Simulation/Simulation.h>

void Simulation::cpuForces(int32 threadId, Particle* particles, int32 nParticles) {
	if(threadId >= nParticles) {
		return;
	}

	Particle& particle = particles[threadId];

	glm::vec3 force(0.f);
	
	for(int32 i = 0; i < particle.NumberOfNeighbors; ++i) {
		Particle::Neighbor const& neighbor = particle.Neighbors[i];
		
		// pressure force
		force += (
			((particle.Position - neighbor.NeighborParticle->Position) / neighbor.Distance) *
			((particle.Pressure + neighbor.NeighborParticle->Pressure) / (2.f * neighbor.NeighborParticle->Density)) *
			utils::powf(config::simulation::physics::smoothingLength - neighbor.Distance, 2)
			) * config::simulation::physics::magicConstants::smoothingKernelNormalizationPressureToForceConstant;
	
		// viscous force
		force += (
			-
			((particle.Velocity - neighbor.NeighborParticle->Velocity) / neighbor.NeighborParticle->Density) * 
			(config::simulation::physics::smoothingLength - neighbor.Distance)
			) * config::simulation::physics::magicConstants::smoothingKernelNormalizationViscousForceConstant;
	}

	// gravity force
	force += glm::vec3(0.f, config::simulation::physics::gravityForce, 0.f);

	particle.Force = force;

}