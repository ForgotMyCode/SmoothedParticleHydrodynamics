#include <Simulation/Simulation.h>

void Simulation::cpuFillGridWithParticles(int32 threadId, Particle* particles, int32 nParticles, Grid grid, int32 particlesPerThread) {

	for(int32 particleIdx = threadId * particlesPerThread; particleIdx < std::min(nParticles, threadId * particlesPerThread + particlesPerThread); ++particleIdx) {
		Particle* particle = particles + threadId;

		auto const& cellIndexs = particle->CellIdx;

		Cell& cell = grid[cellIndexs.z][cellIndexs.y][cellIndexs.x];

		int32 index = cell.FreeIndex.fetch_add(1);

		cell.Particles[index] = particle;
	}

}