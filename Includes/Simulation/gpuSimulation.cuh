/*****************************************************************//**
 * \file   gpuSimulation.cuh
 * \brief  Functions to work with kernels so that the actual GPU stuff is abstracted away.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <niceIntTypes.h>
#include <Simulation/Particle.h>
#include <cudaCheck.h>
#include <cudaargs.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cudaCompatibleConfig.h>

namespace gpuSimulation {

	struct GpuCell {
		Particle** Particles;
		int32 Size;
	};

	/**
	 * \brief Initialize, first step. Should be called before anything else.
	 * \param gpuNeighborBuffer Pointer will be stored here, memory will be allocated to store sufficient amount neghbors.
	 * \param maxNumberOfParticles Max amount of particles in the system.
	 */
	void gpuInitBefore(Particle::Neighbor*& gpuNeighborBuffer, int64 maxNumberOfParticles);

	/**
	 * \brief Initialize, second step. Should be called before anything else, but after gpuInitBefore. It is expected particles are initialized.
	 * \param cpuParticles Initialized particles in host's memory
	 * \param xSamples number of grid cells on x axis
	 * \param ySamples number of grid cells on y axis
	 * \param zSamples number of grid cells on z axis
	 * \param gpuParticles Pointer to memory in device will be stored here, it will be filled with particles.
	 * \param grid Pointer to memory in device will be stored here, sufficiently large to store the grid.
	 * \param gpuGridParticleBuffer Pointer to memory in device will be stored here, sufficiently large to store all particle buffers.
	 * \param gpuDefaultGrid Pointer to memory in device will be stored here. Cleared grid will be stored there.
	 * \param maxNumberOfParticles Max amount of particles in the system.
	 */
	void gpuInitAfter(Particle* cpuParticles, int32 numberOfParticles, int32 xSamples, int32 ySamples, int32 zSamples, Particle*& gpuParticles, GpuCell*& grid, Particle**& gpuGridParticleBuffer, GpuCell*& gpuDefaultGrid, int64 maxNumberOfParticles);

	/**
	 * \brief DO NOT CALL FROM OUTSIDE! GPU kernel to fill grid with particles.
	 * \param particles Buffer of the particle system.
	 * \param nParticles Number of particles in the system.
	 * \param grid Particle grid.
	 */
	__global__ void gpuFillGridWithParticles(Particle* particles, int32 nParticles, GpuCell* grid);

	/**
	 * \brief DO NOT CALL FROM OUTSIDE! GPU kernel to find neighbors and compute their density and pressure.
	 * \param particles Buffer of the particle system.
	 * \param nParticles Number of particles in the system.
	 * \param grid Particle grid.
	 * \param smoothingKernelNormalizationDistanceToDensityConstant Magic precalculated constant from config.
	 * \param particleMass Particle mass.
	 * \param gasConstantK Magic constant K of gas from config.
	 * \param restDensity Rest density of the particle system.
	 */
	__global__ void gpuNeighborsDensityPressure(Particle* particles, int32 nParticles, GpuCell* grid, float smoothingKernelNormalizationDistanceToDensityConstant, float particleMass, float gasConstantK, float restDensity);

	/**
	 * \brief DO NOT CALL FROM OUTSIDE! GPU kernel to calculate forces in the particle system.
	 * \param particles Buffer of the particle system.
	 * \param nParticles Number of particles in the system.
	 * \param smoothingKernelNormalizationPressureToForceConstant Magic precalculated constant from config.
	 * \param particleMass Particle mass.
	 * \param smoothingKernelNormalizationViscousForceConstant Magic precalculated constant from config.
	 * \param gravityConstant Gravity constant. Scales gravity direction.
	 * \param gravityDirection Gravity direction in radians. (One axis is ingored)
	 */
	__global__ void gpuForces(Particle* particles, int32 nParticles, float smoothingKernelNormalizationPressureToForceConstant, float particleMass, float smoothingKernelNormalizationViscousForceConstant, float dynamicViscosity, float gravityConstant, float gravityDirection);

	/**
	 * \brief DO NOT CALL FROM OUTSIDE! GPU kernel to calculate forces in the particle system.
	 * \param particles Buffer of the particle system.
	 * \param nParticles Number of particles in the system.
	 * \param deltaTimeSec Delta time in seconds.
	 * \param outOfBoundsVelocityScale Velocity scale when a particle obunces on a wall of the bounding box.
	 */
	__global__ void gpuVelocities(Particle* particles, int32 nParticles, float deltaTimeSec, float outOfBoundsVelocityScale);

	/**
	 * \brief Invokes GPU kernel to fill grid with particles.
	 * \param gridSize Size of the GPU grid.
	 * \param blockSize Size of the GPU block.
	 * \param particles Buffer of the particle system.
	 * \param nParticles Number of particles in the system.
	 * \param grid Particle grid.
	 */
	void callKernel_gpuFillGridWithParticles(int32 gridSize, int32 blockSize, Particle* particles, int32 nParticles, GpuCell* grid);

	/**
	 * \brief Invokes GPU kernel to find neighbors and compute their density and pressure.
	 * \param gridSize Size of the GPU grid.
	 * \param blockSize Size of the GPU block.
	 * \param particles Buffer of the particle system.
	 * \param nParticles Number of particles in the system.
	 * \param grid Particle grid.
	 * \param smoothingKernelNormalizationDistanceToDensityConstant Magic precalculated constant from config.
	 * \param particleMass Particle mass.
	 * \param gasConstantK Magic constant K of gas from config.
	 * \param restDensity Rest density of the particle system.
	 */
	void callKernel_gpuNeighborsDensityPressure(int32 gridSize, int32 blockSize, Particle* particles, int32 nParticles, GpuCell* grid, float smoothingKernelNormalizationDistanceToDensityConstant, float particleMass, float gasConstantK, float restDensity);

	/**
	 * \brief Invokes GPU kernel to calculate forces in the particle system.
	 * \param gridSize Size of the GPU grid.
	 * \param blockSize Size of the GPU block.
	 * \param particles Buffer of the particle system.
	 * \param nParticles Number of particles in the system.
	 * \param smoothingKernelNormalizationPressureToForceConstant Magic precalculated constant from config.
	 * \param particleMass Particle mass.
	 * \param smoothingKernelNormalizationViscousForceConstant Magic precalculated constant from config.
	 * \param gravityConstant Gravity constant. Scales gravity direction.
	 * \param gravityDirection Gravity direction in radians. (One axis is ingored)
	 */
	void callKernel_gpuForces(int32 gridSize, int32 blockSize, Particle* particles, int32 nParticles, float smoothingKernelNormalizationPressureToForceConstant, float particleMass, float smoothingKernelNormalizationViscousForceConstant, float dynamicViscosity, float gravityConstant, float gravityDirection);

	/**
	 * \brief Invokes GPU kernel to calculate velocities in the particle system and push the particles.
	 * \param gridSize Size of the GPU grid.
	 * \param blockSize Size of the GPU block.
	 * \param particles Buffer of the particle system.
	 * \param nParticles Number of particles in the system.
	 * \param deltaTimeSec Delta time in seconds.
	 * \param outOfBoundsVelocityScale Velocity scale when a particle obunces on a wall of the bounding box.
	 */
	void callKernel_gpuVelocities(int32 gridSize, int32 blockSize, Particle* particles, int32 nParticles, float deltaTimeSec, float outOfBoundsVelocityScale);

	/**
	 * \brief Does necessary operations before calculations (clears the grid). Call at the beginning of every step.
	 * \param grid GPU grid.
	 * \param defaultGrid Cleared GPU grid.
	 * \param xSamples number of grid cells on x axis
	 * \param ySamples number of grid cells on y axis
	 * \param zSamples number of grid cells on z axis
	 */
	void gpuBeginStep(GpuCell* grid, GpuCell* defaultGrid, int32 xSamples, int32 ySamples, int32 zSamples);

	/**
	 * \brief Copies data from device to host. Call at the end of calculation (not necessarily every step).
	 * \param hostParticles Host's particle buffer.
	 * \param devParticles Device's particle buffer.
	 * \param nParticles Number of particles.
	 */
	void gpuFinish(Particle* hostParticles, Particle* devParticles, int32 nParticles);

	/**
	 * \brief Cleans up. Call at the very end.
	 */
	void gpuExit();

	/**
	 * \brief Non-constexpr variant of isCloseToZero so that cudacc can work with it.
	 *  
	 * \param x Number.
	 * \param tolerance Max distance from 0 that is still considered 0.
	 * \return true if x can be considered 0.
	 */
	inline
	__device__ bool isCloseToZero_gpu(float x, float tolerance = 0.000001f) {
		return x < tolerance && x > -tolerance;
	}

}
