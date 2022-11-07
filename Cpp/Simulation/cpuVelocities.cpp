#include <Simulation/Simulation.h>

void Simulation::cpuVelocities(int32 threadId, Particle* particles, int32 nParticles, float deltaTimeSec) {
	if(threadId >= nParticles) {
		return;
	}

	Particle& particle = particles[threadId];

	// time step

	particle.Velocity += utils::isCloseToZero(particle.Density) ? glm::vec3(0.f) : (particle.Force * deltaTimeSec / particle.Density);

	particle.Position += deltaTimeSec * particle.Velocity;

	// bounding box check

	bool const isOut = (
		particle.Position.x <= config::simulation::boundingBox::minX ||
		particle.Position.y <= config::simulation::boundingBox::minY ||
		particle.Position.z <= config::simulation::boundingBox::minZ ||
		particle.Position.x >= config::simulation::boundingBox::maxX ||
		particle.Position.y >= config::simulation::boundingBox::maxY ||
		particle.Position.z >= config::simulation::boundingBox::maxZ
		);

	particle.Position = glm::clamp(particle.Position,
		config::simulation::boundingBox::mins,
		config::simulation::boundingBox::maxs
		);

	if(isOut) {
		particle.Velocity *= config::simulation::physics::outOfBoundsVelocityScale;

		// send the particle little bit towards the center

		
		glm::vec3 center = (config::simulation::boundingBox::maxs + config::simulation::boundingBox::mins) / 2.f;

		glm::vec3 directionToCenter = glm::normalize(center - particle.Position);

		particle.Velocity += directionToCenter * 0.1f;
		
	}
}