#include <Simulation/Simulation.h>

void Simulation::cpuForces(int32 threadId, Particle* particles, int32 nParticles) {
	if(threadId >= nParticles) {
		return;
	}

	Particle& particle = particles[threadId];

	glm::vec3 force(0.f);
	
	for(int32 i = 0; i < particle.NumberOfNeighbors; ++i) {
		Particle::Neighbor const& neighbor = particle.Neighbors[i];

		auto const direction = utils::isCloseToZero(neighbor.Distance) ?
			[]() {
				float const theta = utils::random(0.f, std::numbers::pi_v<float> * 2.f);
				float const phi = utils::random(0.f, std::numbers::pi_v<float> * 2.f);

				float const sinTheta = std::sinf(theta);
				float const cosTheta = std::cosf(theta);

				float const sinPhi = std::sinf(phi);
				float const cosPhi = std::cosf(phi);

				return glm::vec3(
					sinTheta * cosPhi,
					sinTheta * sinPhi,
					cosTheta
				);
			}() 
			:
			(particle.Position - neighbor.NeighborParticle->Position) / neighbor.Distance;
		
		check(neighbor.Distance <= config::simulation::physics::smoothingLength);
		check(neighbor.NeighborParticle->Density > 0.f);

		// pressure force
		force += (
			// vector
			direction *

			// scalar
			(
				((particle.Pressure + neighbor.NeighborParticle->Pressure) / (2.f * neighbor.NeighborParticle->Density)) *
				utils::cube(config::simulation::physics::smoothingLength - neighbor.Distance) *
				config::simulation::physics::magicConstants::smoothingKernelNormalizationPressureToForceConstant * 
				config::simulation::physics::particleMass
			)
			);

		// viscous force
		force += (
			// vector
			((neighbor.NeighborParticle->Velocity - particle.Velocity) / neighbor.NeighborParticle->Density) * 

			// scalar
			(
				(config::simulation::physics::smoothingLength - neighbor.Distance) *
				config::simulation::physics::magicConstants::smoothingKernelNormalizationViscousForceConstant *
				config::simulation::physics::particleMass *
				config::simulation::physics::dynamicViscosity
			)
			) ;	

	}

	// gravity force
	force += glm::vec3(0.f, -config::simulation::physics::particleMass * config::simulation::physics::gravityConstant, 0.f);

	particle.Force += force;

}