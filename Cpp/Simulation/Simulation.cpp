#include <Simulation/Simulation.h>

#include <omp.h>

void Simulation::Initialize() {
	for(int32 i = 0; i < NumberOfParticles; ++i) {
		Particle& particle = Particles[i];

		particle.Position = glm::vec3(
			utils::random(config::simulation::boundingBox::minX, config::simulation::boundingBox::maxX),
			utils::random(config::simulation::boundingBox::minY, config::simulation::boundingBox::maxY),
			utils::random(config::simulation::boundingBox::minZ, config::simulation::boundingBox::maxZ)
		);

		particle.Velocity = glm::vec3(
			utils::random(-0.1, 0.1),
			utils::random(-0.1, 0.1),
			utils::random(-0.1, 0.1)
		);

		//particle.Position = glm::vec3(10.f + float(i), 10.f, 0.f);
		//particle.Velocity = glm::vec3(10.f + float(i), 10.f, 0.f);
	}
}

void Simulation::cpuStepSerial(float deltaTimeSec) {
	//deltaTimeSec = 0.01f;
	glm::clamp(deltaTimeSec, 0.001f, 0.1f);

	ReallocateGridPointer = 0;
	NeighborsDensityPressurePointer = 0;

	FOR_EACH_CELL_ZYX(z, y, x) {
		CellCounts[z][y][x] = 0;
	}

	static constexpr int32 minThreads = std::max(
		config::simulation::maxNumberOfParticles,
		config::simulation::boundingBox::nCells
	);

	for(int32 threadId = 0; threadId < minThreads; ++threadId) {
		cpuCellCounts(threadId, Particles.get(), CellCounts, NumberOfParticles, 1);
	}

	for(int32 threadId = 0; threadId < minThreads; ++threadId) {
		cpuReallocateGrid(threadId, CellCounts, SimulationGrid, GridParticleBuffer.get(), ReallocateGridPointer);
	}

	for(int32 threadId = 0; threadId < minThreads; ++threadId) {
		cpuFillGridWithParticles(threadId, Particles.get(), NumberOfParticles, SimulationGrid, 1);
	}

	for(int32 threadId = 0; threadId < minThreads; ++threadId) {
		cpuNeighborsDensityPressure(threadId, Particles.get(), NumberOfParticles, SimulationGrid, NeighborBuffer.get(), NeighborsDensityPressurePointer);
	}

	for(int32 threadId = 0; threadId < minThreads; ++threadId) {
		cpuForces(threadId, Particles.get(), NumberOfParticles);
	}

	for(int32 threadId = 0; threadId < minThreads; ++threadId) {
		cpuVelocities(threadId, Particles.get(), NumberOfParticles, deltaTimeSec);
	}
}

void Simulation::cpuStepParallel(float deltaTimeSec) {
	//deltaTimeSec = 0.01f;
	glm::clamp(deltaTimeSec, 0.001f, 0.1f);

	ReallocateGridPointer = 0;
	NeighborsDensityPressurePointer = 0;

	FOR_EACH_CELL_ZYX(z, y, x) {
		CellCounts[z][y][x] = 0;
	}

	static constexpr int32 minThreads = std::max(
		config::simulation::maxNumberOfParticles,
		config::simulation::boundingBox::nCells
	);

#pragma omp parallel
	{

#pragma omp for
		for(int32 threadId = 0; threadId < minThreads; ++threadId) {
			cpuCellCounts(threadId, Particles.get(), CellCounts, NumberOfParticles, 1);
		}

#pragma omp barrier
#pragma omp for
		for(int32 threadId = 0; threadId < minThreads; ++threadId) {
			cpuReallocateGrid(threadId, CellCounts, SimulationGrid, GridParticleBuffer.get(), ReallocateGridPointer);
		}
		
#pragma omp barrier
#pragma omp for
		for(int32 threadId = 0; threadId < minThreads; ++threadId) {
			cpuFillGridWithParticles(threadId, Particles.get(), NumberOfParticles, SimulationGrid, 1);
		}
		
#pragma omp barrier
#pragma omp for
		for(int32 threadId = 0; threadId < minThreads; ++threadId) {
			cpuNeighborsDensityPressure(threadId, Particles.get(), NumberOfParticles, SimulationGrid, NeighborBuffer.get(), NeighborsDensityPressurePointer);
		}
		
#pragma omp barrier
#pragma omp for
		for(int32 threadId = 0; threadId < minThreads; ++threadId) {
			cpuForces(threadId, Particles.get(), NumberOfParticles);
		}
		
#pragma omp barrier
#pragma omp for
		for(int32 threadId = 0; threadId < minThreads; ++threadId) {
			cpuVelocities(threadId, Particles.get(), NumberOfParticles, deltaTimeSec);
		}
	}
}