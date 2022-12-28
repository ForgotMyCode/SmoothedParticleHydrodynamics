#include <Simulation/gpuSimulation.cuh>


namespace gpuSimulation {

	__global__ void gpuVelocities(Particle* particles, int32 nParticles, float deltaTimeSec, float outOfBoundsVelocityScale) {
		int32 const threadId = blockDim.x * blockIdx.x + threadIdx.x;

		if(threadId >= nParticles) {
			return;
		}

		Particle& particle = particles[threadId];

		// time step

		particle.Velocity += isCloseToZero_gpu(particle.Density) ? glm::vec3(0.f) : (particle.Force * deltaTimeSec / particle.Density);

		particle.Position += deltaTimeSec * particle.Velocity;

		// bounding box check
		bool isOut = false;

		glm::vec3 mins(
			config::simulation::boundingBox::minX,
			config::simulation::boundingBox::minY,
			config::simulation::boundingBox::minZ
		);

		glm::vec3 maxs(
			config::simulation::boundingBox::maxX,
			config::simulation::boundingBox::maxY,
			config::simulation::boundingBox::maxZ
			);

		for(int dim = 0; dim < 3; ++dim) {
			if(particle.Position[dim] < mins[dim] || particle.Position[dim] > maxs[dim]) {
				isOut = true;

				particle.Velocity[dim] *= -1.f;
				particle.Velocity *= outOfBoundsVelocityScale;
			}
		}

		if(isOut) {
			particle.Position = glm::clamp(particle.Position,
				mins,
				maxs
			);
		}
	}
}