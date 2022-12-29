/*****************************************************************//**
 * \file   Simulation.h
 * \brief  Particle simulation - computes the SPH.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

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
		// Buffer of particles in this cell
		Particle** Particles{};

		// How many particles are in this cell
		std::atomic<int32> Size{};
	};

	template<typename T>
	using Cells = std::array<std::array<std::array<T, config::simulation::boundingBox::xSamples>, config::simulation::boundingBox::ySamples>, config::simulation::boundingBox::zSamples>;
	
	// Host grid
	using Grid = Cells<Cell>;

	/**
	 * \brief Initialize the simulation. Should be called before anything else.
	 * 
	 */
	void Initialize();

	/**
	 * \brief Do a completely serial step of the simulation on CPU.
	 * 
	 * \param deltaTimeSec Delta time in seconds.
	 */
	void cpuStepSerial(float deltaTimeSec);

	/**
	 * \brief Do a parallel step of the simulation on CPU.
	 * 
	 * \param deltaTimeSec Delta time in seconds.
	 */
	void cpuStepParallel(float deltaTimeSec);

	/**
	 * \brief Do a parallel step of the simulation on GPU..
	 * 
	 * \param deltaTimeSec Delta time in seconds.
	 */
	void gpuStepParallel(float deltaTimeSec);

	/**
	 * \brief Fill grid with particles.
	 * 
	 * \param threadId Thread ID. May or may not be a literal thread.
	 * \param particles Buffer of the particle system.
	 * \param nParticles Number of particles in the system.
	 * \param grid Particle grid.
	 * \param particlesPerThread deprecated, use 1
	 */
	void cpuFillGridWithParticles(int32 threadId, Particle* particles, int32 nParticles, Grid& grid, int32 particlesPerThread);

	/**
	 * \brief Find neighbors and compute their density and pressure.
	 * 
	 * \param threadId Thread ID. May or may not be a literal thread.
	 * \param particles Buffer of the particle system.
	 * \param nParticles Number of particles in the system.
	 * \param grid Particle grid.
	 */
	void cpuNeighborsDensityPressure(int32 threadId, Particle* particles, int32 nParticles, Grid& grid);

	/**
	 * \brief Calculate forces in the particle system.
	 * 
	 * \param threadId Thread ID. May or may not be a literal thread.
	 * \param particles Buffer of the particle system.
	 * \param nParticles Number of particles in the system.
	 */
	void cpuForces(int32 threadId, Particle* particles, int32 nParticles);

	/**
	 * \brief Calculate velocities in the particle system and push the particles.
	 * 
	 * \param threadId Thread ID. May or may not be a literal thread.
	 * \param particles Buffer of the particle system.
	 * \param nParticles Number of particles in the system.
	 * \param deltaTimeSec Delta time in seconds.
	 */
	void cpuVelocities(int32 threadId, Particle* particles, int32 nParticles, float deltaTimeSec);

	/**
	 * \brief Maps index of a flattened 3d array to x, y, z indices.
	 * 
	 * \param index Flattened index.
	 * \return vec3 3d index.
	 */
	static auto map1dIndexTo3dCell(int32 index) {
		const int32 z = index / (config::simulation::boundingBox::ySamples * config::simulation::boundingBox::xSamples);
		index -= z * (config::simulation::boundingBox::ySamples * config::simulation::boundingBox::xSamples);

		const int32 y = index / config::simulation::boundingBox::xSamples;
		index -= y * config::simulation::boundingBox::xSamples;

		const int32 x = index;

		return glm::vec<3, int32>(x, y, z);
	}

	/**
	 * \brief Get the number of particles in this particle system.
	 * 
	 * \return Number of particles.
	 */
	int32 GetNumberOfParticles() const {
		return this->NumberOfParticles;
	}

	/**
	 * \brief Get particle buffer of this simulation.
	 * 
	 * \return Particle buffer.
	 */
	Particle* GetParticles() {
		return Particles.get();
	}

private:
	// Particle buffer on host's side
	std::unique_ptr<Particle[]> Particles;

	// Host's grid, !! must be cleared each step (if cpu)!!
	std::unique_ptr<Grid> SimulationGrid;

	// Buffer for particle buffers, split in grid (if cpu)
	std::unique_ptr<Particle*[]> GridParticleBuffer;

	// Buffer for neighbors (if cpu)
	std::unique_ptr<Particle::Neighbor[]> NeighborBuffer;

	// Pointer to device's grid (if gpu)
	gpuSimulation::GpuCell* gpuGrid{};

	// Does not fit in constant memory :(
	// Pointer to device's cleared grid (if gpu)
	gpuSimulation::GpuCell* gpuDefaultGrid{};

	// Device's particle buffer (if gpu)
	Particle* gpuParticles{};

	// Device's buffer for particle buffers (if gpu)
	Particle** gpuGridParticleBuffer{};

	// Device's buffer for neighbors (if gpu)
	Particle::Neighbor* gpuNeighborBuffer{};

	// number of particles, I've eventually decided to make it static
	int32 NumberOfParticles = config::simulation::maxNumberOfParticles;
};