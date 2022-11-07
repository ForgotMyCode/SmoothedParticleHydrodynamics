#pragma once

#include <array>
#include <atomic>
#include <glm/glm.hpp>
#include <memory>

#include <core.h>

struct Simulation {

	struct Particle {
		struct Neighbor {
			Particle* NeighborParticle{};
			float Distance;
		};

		// position of the particle
		glm::vec3 Position{};

		// velocity of the particle
		glm::vec3 Velocity{};

		// accumulated forces acting upon the particle
		glm::vec3 Force{};

		// index of cell in the grid this particle belongs to
		glm::vec<3, int32> CellIdx{};

		// neighbors of this particle
		Neighbor* Neighbors{};

		// # of neighbors
		int32 NumberOfNeighbors{};

		float Pressure{};

		float Density{};
	};

	struct Cell {
		Particle** Particles{};

		int32 Size{};

		std::atomic<int32> FreeIndex{};
	};

	template<typename T>
	using Cells = T[config::simulation::boundingBox::zSamples][config::simulation::boundingBox::ySamples][config::simulation::boundingBox::xSamples];
	
	using Grid = Cells<Cell>;

	void Initialize();

	void cpuStepSerial(float deltaTimeSec);

	void cpuStepParallel(float deltaTimeSec);

	void cpuCellCounts(int32 threadId, Particle* particles, Cells<std::atomic<int32>> cellCounts, int32 nParticles, int32 particlesPerThread);

	void cpuReallocateGrid(int32 threadId, Cells<std::atomic<int32>> cellCounts, Grid grid, Particle** memoryBuffer, std::atomic<int32>& blockPointer);

	void cpuFillGridWithParticles(int32 threadId, Particle* particles, int32 nParticles, Grid grid, int32 particlesPerThread);

	void cpuNeighborsDensityPressure(int32 threadId, Particle* particles, int32 nParticles, Grid grid, Particle::Neighbor* memoryBuffer, std::atomic<int32>& blockPointer);

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
	// !! must be cleared each step !!
	std::atomic<int32> ReallocateGridPointer{}, NeighborsDensityPressurePointer{};

	std::unique_ptr<Particle[]> Particles = std::make_unique<Particle[]>(config::simulation::maxNumberOfParticles);

	// !! must be cleared each step !!
	Cells<std::atomic<int32>> CellCounts;

	Grid SimulationGrid;

	std::unique_ptr<Particle*[]> GridParticleBuffer = std::make_unique<Particle*[]>(config::simulation::maxNumberOfParticles);

	std::unique_ptr<Particle::Neighbor[]> NeighborBuffer = std::make_unique<Particle::Neighbor[]>(config::simulation::maxNumberOfParticles * config::simulation::maxNumberOfParticles);



	int32 NumberOfParticles = config::simulation::maxNumberOfParticles;
};