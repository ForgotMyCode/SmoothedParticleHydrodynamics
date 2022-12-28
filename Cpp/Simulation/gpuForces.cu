#include <Simulation/gpuSimulation.cuh>

#include <curand_kernel.h>
#include <curand.h>

namespace gpuSimulation {

	__global__ void gpuForces(Particle* particles, int32 nParticles, float smoothingKernelNormalizationPressureToForceConstant, float particleMass, float smoothingKernelNormalizationViscousForceConstant, float dynamicViscosity, float gravityConstant, float gravityDirection) {
		
		int32 const threadId = blockDim.x * blockIdx.x + threadIdx.x;

		if(threadId >= nParticles) {
			return;
		}

		Particle& particle = particles[threadId];

		glm::vec3 force(0.f);

		for(int32 i = 0; i < particle.NumberOfNeighbors; ++i) {
			Particle::Neighbor const& neighbor = particle.Neighbors[i];

			glm::vec3 direction{};

			if(isCloseToZero_gpu(neighbor.Distance)) {
				curandState state;
				curand_init(uint64(clock() + threadId), 0, 0, &state);

				float const theta = curand_uniform(&state) * PI_GPU * 2.f;
				float const phi = curand_uniform(&state) * PI_GPU * 2.f;
			
				float const sinTheta = std::sinf(theta);
				float const cosTheta = std::cosf(theta);

				float const sinPhi = std::sinf(phi);
				float const cosPhi = std::cosf(phi);

				direction.x = sinTheta * cosPhi;
				direction.y = sinTheta * sinPhi;
				direction.z = cosTheta;
			}
			else {
				direction = (particle.Position - neighbor.NeighborParticle->Position) / neighbor.Distance;
			}

			auto const diff = config::simulation::physics::smoothingLength - neighbor.Distance;

			// pressure force
			force += (
				// vector
				direction *

				// scalar
				(
					((particle.Pressure + neighbor.NeighborParticle->Pressure) / (2.f * neighbor.NeighborParticle->Density)) *
					CUBE(diff) *
					smoothingKernelNormalizationPressureToForceConstant *
					particleMass
					)
				);

			// viscous force
			force += (
				// vector
				((neighbor.NeighborParticle->Velocity - particle.Velocity) / neighbor.NeighborParticle->Density) *

				// scalar
				(
					(config::simulation::physics::smoothingLength - neighbor.Distance) *
					smoothingKernelNormalizationViscousForceConstant *
					particleMass *
					dynamicViscosity
					)
				);

		}

		// gravity force
		force += particleMass * gravityConstant * glm::vec3(std::sinf(gravityDirection), -std::cosf(gravityDirection), 0.f);

		particle.Force = force;

	}
}