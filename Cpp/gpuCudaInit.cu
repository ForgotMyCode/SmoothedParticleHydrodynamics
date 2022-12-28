#include <Simulation/gpuSimulation.cuh>

namespace gpuSimulation {
	
	void gpuInitBefore(Particle::Neighbor*& gpuNeighborBuffer, int64 maxNumberOfParticles) {

		CUDA_CHECK_ERROR(cudaMalloc(&gpuNeighborBuffer, sizeof(Particle::Neighbor) * maxNumberOfParticles * maxNumberOfParticles));
		
	}

	void gpuInitAfter(Particle* cpuParticles, int32 numberOfParticles, int32 xSamples, int32 ySamples, int32 zSamples, Particle*& gpuParticles, GpuCell*& grid, Particle**& gpuGridParticleBuffer, GpuCell*& gpuDefaultGrid, int64 maxNumberOfParticles) {
		// TODO

		CUDA_CHECK_ERROR(cudaMalloc(&gpuParticles, numberOfParticles * sizeof(Particle)));

		CUDA_CHECK_ERROR(cudaMemcpy(gpuParticles, cpuParticles, numberOfParticles * sizeof(Particle), cudaMemcpyHostToDevice));

		CUDA_CHECK_ERROR(cudaMalloc(&gpuGridParticleBuffer, xSamples * ySamples * zSamples * maxNumberOfParticles * sizeof(Particle*)));

		CUDA_CHECK_ERROR(cudaMalloc(&grid, sizeof(GpuCell) * xSamples * ySamples * zSamples));

		CUDA_CHECK_ERROR(cudaMalloc(&gpuDefaultGrid, sizeof(GpuCell) * xSamples * ySamples * zSamples));

		int64 offset = 0;

		GpuCell cell{};

		for(int64 i = 0; i < xSamples * ySamples * zSamples; ++i) {

			cell.Particles = gpuGridParticleBuffer + offset;

			CUDA_CHECK_ERROR(cudaMemcpy((void*) (gpuDefaultGrid + i), (void const*) &cell, sizeof(GpuCell), cudaMemcpyHostToDevice));

			CUDA_CHECK_ERROR(cudaDeviceSynchronize());

			offset += maxNumberOfParticles;
		}
	}

	void gpuBeginStep(GpuCell* grid, GpuCell* defaultGrid, int32 xSamples, int32 ySamples, int32 zSamples) {

		// reset the grid
		CUDA_CHECK_ERROR(cudaMemcpy(grid, defaultGrid, sizeof(GpuCell) * xSamples * ySamples * zSamples, cudaMemcpyDeviceToDevice));

	}

	void gpuFinishStep(Particle* hostParticles, Particle* devParticles, int32 nParticles) {
		CUDA_CHECK_ERROR(cudaMemcpy(hostParticles, devParticles, sizeof(Particle) * nParticles, cudaMemcpyDeviceToHost));
	}

	void gpuExit() {
		// TODO
	}

	void callKernel_gpuFillGridWithParticles(int32 gridSize, int32 blockSize, Particle* particles, int32 nParticles, GpuCell* grid) {
		gpuFillGridWithParticles CUDA_ARGS(gridSize, blockSize) (particles, nParticles, grid);
	}

	void callKernel_gpuNeighborsDensityPressure(int32 gridSize, int32 blockSize, Particle* particles, int32 nParticles, GpuCell* grid, float smoothingKernelNormalizationDistanceToDensityConstant, float particleMass, float gasConstantK, float restDensity) {
		gpuNeighborsDensityPressure CUDA_ARGS(gridSize, blockSize) (particles, nParticles, grid, smoothingKernelNormalizationDistanceToDensityConstant, particleMass, gasConstantK, restDensity);
	
	}

	void callKernel_gpuForces(int32 gridSize, int32 blockSize, Particle* particles, int32 nParticles, float smoothingKernelNormalizationPressureToForceConstant, float particleMass, float smoothingKernelNormalizationViscousForceConstant, float dynamicViscosity, float gravityConstant, float gravityDirection) {
		gpuForces CUDA_ARGS(gridSize, blockSize) (particles, nParticles, smoothingKernelNormalizationPressureToForceConstant, particleMass, smoothingKernelNormalizationViscousForceConstant, dynamicViscosity, gravityConstant, gravityDirection);
	}

	void callKernel_gpuVelocities(int32 gridSize, int32 blockSize, Particle* particles, int32 nParticles, float deltaTimeSec, float outOfBoundsVelocityScale) {
		gpuVelocities CUDA_ARGS(gridSize, blockSize) (particles, nParticles, deltaTimeSec, outOfBoundsVelocityScale);
	}

}