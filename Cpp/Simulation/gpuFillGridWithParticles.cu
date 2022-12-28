#include <Simulation/gpuSimulation.cuh>

namespace gpuSimulation {

	__global__ void gpuFillGridWithParticles(Particle* particles, int32 nParticles, GpuCell* grid) {

		int32 const threadId = blockDim.x * blockIdx.x + threadIdx.x;

		if(threadId >= nParticles) {
			return;
		}

		Particle& particle = particles[threadId];

		auto& cellIndexs = particle.CellIdx;

		cellIndexs.x = glm::clamp(int32((particle.Position.x - config::simulation::boundingBox::minX) / config::simulation::physics::smoothingLength),
			0, config::simulation::boundingBox::xSamples - 1);

		cellIndexs.y = glm::clamp(int32((particle.Position.y - config::simulation::boundingBox::minY) / config::simulation::physics::smoothingLength),
			0, config::simulation::boundingBox::ySamples - 1);

		cellIndexs.z = glm::clamp(int32((particle.Position.z - config::simulation::boundingBox::minZ) / config::simulation::physics::smoothingLength),
			0, config::simulation::boundingBox::zSamples - 1);

		GpuCell& cell = grid[FLATTEN_3D_INDEX(cellIndexs.x, cellIndexs.y, cellIndexs.z, config::simulation::boundingBox::xSamples, config::simulation::boundingBox::ySamples)];

		int32 index = atomicAdd(&cell.Size, 1);

		cell.Particles[index] = &particle;
	}
}