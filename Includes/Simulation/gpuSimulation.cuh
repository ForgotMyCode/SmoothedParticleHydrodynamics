
#include <cudaargs.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

namespace gpuSimulation {

	void cudaInit();

	__global__ void gpuFillGridWithParticles();

	__global__ void gpuNeighborsDensityPressure();

	__global__ void gpuForces();

	__global__ void gpuVelocities();

}
