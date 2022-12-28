#include <Simulation/gpuSimulation.cuh>

#include <array>
#include <cmath>


namespace gpuSimulation {

	struct XYZ {
		int32 x, y, z;
	};

	__global__ void gpuNeighborsDensityPressure(Particle* particles, int32 nParticles, GpuCell* grid, float smoothingKernelNormalizationDistanceToDensityConstant, float particleMass, float gasConstantK, float restDensity) {		

		static constexpr XYZ lookaround[27] = {
		{ -1 , -1 , -1 },
		{ -1 , -1 , 0 },
		{ -1 , -1 , 1 },
		{ -1 , 0 , -1 },
		{ -1 , 0 , 0 },
		{ -1 , 0 , 1 },
		{ -1 , 1 , -1 },
		{ -1 , 1 , 0 },
		{ -1 , 1 , 1 },
		{ 0 , -1 , -1 },
		{ 0 , -1 , 0 },
		{ 0 , -1 , 1 },
		{ 0 , 0 , -1 },
		{ 0 , 0 , 0 },
		{ 0 , 0 , 1 },
		{ 0 , 1 , -1 },
		{ 0 , 1 , 0 },
		{ 0 , 1 , 1 },
		{ 1 , -1 , -1 },
		{ 1 , -1 , 0 },
		{ 1 , -1 , 1 },
		{ 1 , 0 , -1 },
		{ 1 , 0 , 0 },
		{ 1 , 0 , 1 },
		{ 1 , 1 , -1 },
		{ 1 , 1 , 0 },
		{ 1 , 1 , 1 },
		};

		int32 const threadId = blockDim.x * blockIdx.x + threadIdx.x;

		if(threadId >= nParticles) {
			return;
		}

		auto const smoothingLengthSquared = config::simulation::physics::smoothingLength * config::simulation::physics::smoothingLength;

		auto const smoothingLengthSquaredCubed = smoothingLengthSquared * smoothingLengthSquared * smoothingLengthSquared;

		Particle& particle = particles[threadId];

		auto const& cellIdxs = particle.CellIdx;

		int32 const cellX = cellIdxs.x;
		int32 const cellY = cellIdxs.y;
		int32 const cellZ = cellIdxs.z;

		int32& neighborCount = particle.NumberOfNeighbors;
		neighborCount = 0;

		for(int32 i = 0; i < 27; ++i) {
			auto const& off = lookaround[i];
			auto const xOff = off.x;
			auto const yOff = off.y;
			auto const zOff = off.z;

			int32 const xx = cellX + xOff;
			int32 const yy = cellY + yOff;
			int32 const zz = cellZ + zOff;

			if(xx < 0 || yy < 0 || zz < 0 || xx >= config::simulation::boundingBox::xSamples || yy >= config::simulation::boundingBox::ySamples || zz >= config::simulation::boundingBox::zSamples) {
				continue;
			}

			GpuCell& targetCell = grid[FLATTEN_3D_INDEX(xx, yy, zz, config::simulation::boundingBox::xSamples, config::simulation::boundingBox::ySamples)];

			for(int32 i = 0; i < targetCell.Size; ++i) {
				Particle* targetParticle = targetCell.Particles[i];

				auto const direction = targetParticle->Position - particle.Position;

				float distanceSquared = glm::dot(direction, direction);

				if(distanceSquared <= (smoothingLengthSquared) &&
					targetParticle != &particle
					) {
					particle.Neighbors[neighborCount].Distance = std::sqrtf(distanceSquared);
					particle.Neighbors[neighborCount].NeighborParticle = targetParticle;

					++neighborCount;
				}
			}
		}

		float density = (
			smoothingKernelNormalizationDistanceToDensityConstant *
			particleMass *
			CUBE(smoothingLengthSquared)
			);

		for(int32 i = 0; i < neighborCount; ++i) {

			auto const dist = particle.Neighbors[i].Distance;

			auto const diff = smoothingLengthSquared - (dist * dist);
			density += (
				smoothingKernelNormalizationDistanceToDensityConstant *
				particleMass *
				CUBE(diff)
				);
		}

		particle.Density = density;

		particle.Pressure = gasConstantK * (density - restDensity);

	}

}