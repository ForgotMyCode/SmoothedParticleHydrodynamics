#include <Simulation/Simulation.h>

#ifdef DEPRECATED

void Simulation::cpuCellCounts(int32 threadId, Particle* particles, Cells<std::atomic<int32>> cellCounts, int32 nParticles, int32 particlesPerThread) {

	Cells<int32> localCellCounts{};

	for(int32 particleIdx = threadId * particlesPerThread; particleIdx < std::min(nParticles, threadId * particlesPerThread + particlesPerThread); ++particleIdx) {
		Particle& particle = particles[particleIdx];

		particle.CellIdx = glm::clamp(
			glm::vec<3, int32>((particle.Position - config::simulation::boundingBox::mins) / config::simulation::physics::smoothingLength),
			glm::vec<3, int32>(0),
			config::simulation::boundingBox::samples - int32(1)
		);

		++localCellCounts[particle.CellIdx.z][particle.CellIdx.y][particle.CellIdx.x];

		particle.Force = glm::vec3(0.f);
	}

	FOR_EACH_CELL_ZYX(z, y, x) {
		if(localCellCounts[z][y][x] > 0) {
			// atomic add
			cellCounts[z][y][x] += localCellCounts[z][y][x];
		}
	}

}

#endif