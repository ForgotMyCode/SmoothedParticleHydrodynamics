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

	void gpuInitBefore(Particle::Neighbor*& gpuNeighborBuffer, int64 maxNumberOfParticles);

	void gpuInitAfter(Particle* cpuParticles, int32 numberOfParticles, int32 xSamples, int32 ySamples, int32 zSamples, Particle*& gpuParticles, GpuCell*& grid, Particle**& gpuGridParticleBuffer, GpuCell*& gpuDefaultGrid, int64 maxNumberOfParticles);

	__global__ void gpuFillGridWithParticles(Particle* particles, int32 nParticles, GpuCell* grid);

	__global__ void gpuNeighborsDensityPressure(Particle* particles, int32 nParticles, GpuCell* grid, float smoothingKernelNormalizationDistanceToDensityConstant, float particleMass, float gasConstantK, float restDensity);

	__global__ void gpuForces(Particle* particles, int32 nParticles, float smoothingKernelNormalizationPressureToForceConstant, float particleMass, float smoothingKernelNormalizationViscousForceConstant, float dynamicViscosity, float gravityConstant, float gravityDirection);

	__global__ void gpuVelocities(Particle* particles, int32 nParticles, float deltaTimeSec, float outOfBoundsVelocityScale);

	void callKernel_gpuFillGridWithParticles(int32 gridSize, int32 blockSize, Particle* particles, int32 nParticles, GpuCell* grid);

	void callKernel_gpuNeighborsDensityPressure(int32 gridSize, int32 blockSize, Particle* particles, int32 nParticles, GpuCell* grid, float smoothingKernelNormalizationDistanceToDensityConstant, float particleMass, float gasConstantK, float restDensity);

	void callKernel_gpuForces(int32 gridSize, int32 blockSize, Particle* particles, int32 nParticles, float smoothingKernelNormalizationPressureToForceConstant, float particleMass, float smoothingKernelNormalizationViscousForceConstant, float dynamicViscosity, float gravityConstant, float gravityDirection);

	void callKernel_gpuVelocities(int32 gridSize, int32 blockSize, Particle* particles, int32 nParticles, float deltaTimeSec, float outOfBoundsVelocityScale);

	void gpuBeginStep(GpuCell* grid, GpuCell* defaultGrid, int32 xSamples, int32 ySamples, int32 zSamples);

	void gpuFinishStep(Particle* hostParticles, Particle* devParticles, int32 nParticles);

	void gpuExit();

	//__constant__ GpuCell defaultGrid[config::simulation::boundingBox::xSamples * config::simulation::boundingBox::ySamples * config::simulation::boundingBox::zSamples];

	inline
	__device__ bool isCloseToZero_gpu(float x, float tolerance = 0.000001f) {
		return x < tolerance && x > -tolerance;
	}

}
