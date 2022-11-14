#include <Simulation/Simulation.h>

void Simulation::cpuVelocities(int32 threadId, Particle* particles, int32 nParticles, float deltaTimeSec) {
	if(threadId >= nParticles) {
		return;
	}

	Particle& particle = particles[threadId];

	// time step

	check(particle.Density > 0.f);

	particle.Velocity += utils::isCloseToZero(particle.Density) ? glm::vec3(0.f) : (particle.Force * deltaTimeSec / particle.Density);

	particle.Position += deltaTimeSec * particle.Velocity;

	// bounding box check
	bool isOut = false;

	for(int dim = 0; dim < 3; ++dim) {
		if(particle.Position[dim] < config::simulation::boundingBox::mins[dim] || particle.Position[dim] > config::simulation::boundingBox::maxs[dim]) {
			isOut = true;

			particle.Velocity[dim] *= -1.f;
			particle.Velocity *= config::simulation::physics::outOfBoundsVelocityScale;
		}
	}

	if(isOut) {
		particle.Position = glm::clamp(particle.Position,
			config::simulation::boundingBox::mins,
			config::simulation::boundingBox::maxs
		);
	}
}