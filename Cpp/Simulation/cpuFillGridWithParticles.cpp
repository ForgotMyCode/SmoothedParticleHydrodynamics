#include <Simulation/Simulation.h>

void Simulation::cpuFillGridWithParticles(int32 threadId, Particle* particles, int32 nParticles, Grid grid, int32 particlesPerThread) {

	for(int32 particleIdx = threadId * particlesPerThread; particleIdx < std::min(nParticles, threadId * particlesPerThread + particlesPerThread); ++particleIdx) {
		Particle& particle = particles[threadId];

		auto& cellIndexs = particle.CellIdx;

		cellIndexs = glm::clamp(
			glm::vec<3, int32>((particle.Position - config::simulation::boundingBox::mins) / config::simulation::physics::smoothingLength),
			glm::vec<3, int32>(0),
			config::simulation::boundingBox::samples - int32(1)
		);

		Cell& cell = grid[cellIndexs.z][cellIndexs.y][cellIndexs.x];

		int32 index = cell.Size.fetch_add(1);

		cell.Particles[index] = &particle;
	}

}