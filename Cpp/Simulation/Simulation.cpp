#include <Simulation/Simulation.h>

#include <omp.h>

#include <Window.h>

void Simulation::Initialize() {
	 Particles = std::make_unique<Particle[]>(config::simulation::maxNumberOfParticles);

	GridParticleBuffer = std::make_unique<Particle*[]>(config::simulation::maxNumberOfParticles * config::simulation::boundingBox::nCells);

	NeighborBuffer = std::make_unique<Particle::Neighbor[]>(config::simulation::maxNumberOfParticles * config::simulation::maxNumberOfParticles);

	for(int32 i = 0; i < NumberOfParticles; ++i) {
		Particle& particle = Particles[i];

		particle.Position = glm::vec3(
			utils::random(config::simulation::boundingBox::minX / 3.f, config::simulation::boundingBox::maxX / 3.f),
			utils::random(config::simulation::boundingBox::minY / 3.f, config::simulation::boundingBox::maxY / 3.f),
			utils::random(config::simulation::boundingBox::minZ / 3.f, config::simulation::boundingBox::maxZ / 3.f)
		);

		particle.Velocity = glm::vec3(
			utils::random(-0.1f, 0.1f),
			utils::random(-0.1f, 0.1f),
			utils::random(-0.1f, 0.1f)
		);

		particle.Neighbors = this->NeighborBuffer.get() + (i * config::simulation::maxNumberOfParticles);

		//particle.Position = glm::vec3(10.f + float(i), 10.f, 0.f);
		//particle.Velocity = glm::vec3(10.f + float(i), 10.f, 0.f);
	}

	int64 offset = 0;

	FOR_EACH_CELL_ZYX(z, y, x) {
		SimulationGrid[z][y][x].Particles = this->GridParticleBuffer.get() + offset;
		offset += config::simulation::maxNumberOfParticles;
	}
}

void Simulation::cpuStepSerial(float deltaTimeSec) {

	float const timeStart = Window::GetActiveWindow()->GetTimeSeconds();

	//deltaTimeSec = 0.01f;
	deltaTimeSec = glm::clamp(deltaTimeSec, 0.001f, 0.1f);

	FOR_EACH_CELL_ZYX(z, y, x) {
		SimulationGrid[z][y][x].Size = 0;
	}

	static constexpr int32 threadCount = std::max(
		int32(config::simulation::maxNumberOfParticles),
		config::simulation::boundingBox::nCells
	);

	for(int32 threadId = 0; threadId < threadCount; ++threadId) {
		cpuFillGridWithParticles(threadId, Particles.get(), NumberOfParticles, SimulationGrid, 1);
	}

	for(int32 threadId = 0; threadId < threadCount; ++threadId) {
		cpuNeighborsDensityPressure(threadId, Particles.get(), NumberOfParticles, SimulationGrid);
	}

	for(int32 threadId = 0; threadId < threadCount; ++threadId) {
		cpuForces(threadId, Particles.get(), NumberOfParticles);
	}

	for(int32 threadId = 0; threadId < threadCount; ++threadId) {
		cpuVelocities(threadId, Particles.get(), NumberOfParticles, glm::clamp(deltaTimeSec, 0.0001f, 0.1f) * config::simulation::physics::timeScale);
	}
	
	float const timeEnd = Window::GetActiveWindow()->GetTimeSeconds();

	Window::GetActiveWindow()->AddCalculationTime(timeEnd - timeStart);

}

void Simulation::cpuStepParallel(float deltaTimeSec) {	

	float const timeStart = Window::GetActiveWindow()->GetTimeSeconds();

	static constexpr int32 threadCount = std::max(
		int32(config::simulation::maxNumberOfParticles),
		config::simulation::boundingBox::nCells
	);

#pragma omp parallel
	{

#pragma omp for
		FOR_EACH_CELL_ZYX(z, y, x) {
			SimulationGrid[z][y][x].Size = 0;
		}
		
#pragma omp barrier
#pragma omp for
		for(int32 threadId = 0; threadId < threadCount; ++threadId) {
			cpuFillGridWithParticles(threadId, Particles.get(), NumberOfParticles, SimulationGrid, 1);
		}
		
#pragma omp barrier
#pragma omp for
		for(int32 threadId = 0; threadId < threadCount; ++threadId) {
			cpuNeighborsDensityPressure(threadId, Particles.get(), NumberOfParticles, SimulationGrid);
		}
		
#pragma omp barrier
#pragma omp for
		for(int32 threadId = 0; threadId < threadCount; ++threadId) {
			cpuForces(threadId, Particles.get(), NumberOfParticles);
		}
		
#pragma omp barrier
#pragma omp for
		for(int32 threadId = 0; threadId < threadCount; ++threadId) {
			cpuVelocities(threadId, Particles.get(), NumberOfParticles, glm::clamp(deltaTimeSec, 0.0001f, 0.1f) * config::simulation::physics::timeScale);
		}
	}
	
	
	float const timeEnd = Window::GetActiveWindow()->GetTimeSeconds();

	Window::GetActiveWindow()->AddCalculationTime(timeEnd - timeStart);

}