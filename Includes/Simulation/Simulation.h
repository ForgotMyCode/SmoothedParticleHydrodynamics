#pragma once

#include <array>
#include <atomic>
#include <glm/glm.hpp>
#include <memory>

#include <core.h>
#include <Simulation/Particle.h>

namespace gpuSimulation {
	struct GpuCell;
}

struct Simulation {

	struct Cell {
		Particle** Particles{};

		std::atomic<int32> Size{};
	};

	template<typename T>
	using Cells = std::array<std::array<std::array<T, config::simulation::boundingBox::xSamples>, config::simulation::boundingBox::ySamples>, config::simulation::boundingBox::zSamples>;
	
	using Grid = Cells<Cell>;

	void Initialize();

	void cpuStepSerial(float deltaTimeSec);

	void cpuStepParallel(float deltaTimeSec);

	void gpuStepParallel(float deltaTimeSec);

	void cpuFillGridWithParticles(int32 threadId, Particle* particles, int32 nParticles, Grid& grid, int32 particlesPerThread);

	void cpuNeighborsDensityPressure(int32 threadId, Particle* particles, int32 nParticles, Grid& grid);

	void cpuForces(int32 threadId, Particle* particles, int32 nParticles);

	void cpuVelocities(int32 threadId, Particle* particles, int32 nParticles, float deltaTimeSec);

	static auto map1dIndexTo3dCell(int32 index) {
		const int32 z = index / (config::simulation::boundingBox::ySamples * config::simulation::boundingBox::xSamples);
		index -= z * (config::simulation::boundingBox::ySamples * config::simulation::boundingBox::xSamples);

		const int32 y = index / config::simulation::boundingBox::xSamples;
		index -= y * config::simulation::boundingBox::xSamples;

		const int32 x = index;

		return glm::vec<3, int32>(x, y, z);
	}

	int32 GetNumberOfParticles() const {
		return this->NumberOfParticles;
	}

	Particle* GetParticles() {
		return Particles.get();
	}

private:
	std::unique_ptr<Particle[]> Particles;

	// !! must be cleared each step !!
	std::unique_ptr<Grid> SimulationGrid;

	std::unique_ptr<Particle*[]> GridParticleBuffer;

	std::unique_ptr<Particle::Neighbor[]> NeighborBuffer;

	gpuSimulation::GpuCell* gpuGrid{};

	// Does not fit in constant memory :(
	gpuSimulation::GpuCell* gpuDefaultGrid{};

	Particle* gpuParticles{};

	Particle** gpuGridParticleBuffer{};

	Particle::Neighbor* gpuNeighborBuffer{};

	int32 NumberOfParticles = config::simulation::maxNumberOfParticles;
};