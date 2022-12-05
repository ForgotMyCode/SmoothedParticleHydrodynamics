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

		std::atomic<int32> Size{};
	};

	template<typename T>
	using Cells = T[config::simulation::boundingBox::zSamples][config::simulation::boundingBox::ySamples][config::simulation::boundingBox::xSamples];
	
	using Grid = Cells<Cell>;

	void Initialize();

	void cpuStepSerial(float deltaTimeSec);

	void cpuStepParallel(float deltaTimeSec);

	void cpuFillGridWithParticles(int32 threadId, Particle* particles, int32 nParticles, Grid grid, int32 particlesPerThread);

	void cpuNeighborsDensityPressure(int32 threadId, Particle* particles, int32 nParticles, Grid grid);

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
	Grid SimulationGrid;

	std::unique_ptr<Particle*[]> GridParticleBuffer;

	std::unique_ptr<Particle::Neighbor[]> NeighborBuffer;

	Grid* gpuGrid{};

	Particle* gpuParticles{};

	Particle** gpuGridParticleBuffer{};

	Particle::Neighbor* gpuNeighborBuffer{};

	int32 NumberOfParticles = config::simulation::maxNumberOfParticles;
};