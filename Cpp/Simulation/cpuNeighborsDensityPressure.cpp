#include <Simulation/Simulation.h>

#include <cmath>
#include <tuple>

namespace {
	constexpr static std::array<std::tuple<int32, int32, int32>, 27> lookaround{ {
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
		} };
}

void Simulation::cpuNeighborsDensityPressure(int32 threadId, Particle* particles, int32 nParticles, Grid grid, Particle::Neighbor* memoryBuffer, std::atomic<int32>& blockPointer) {
	if(threadId >= nParticles) {
		return;
	}

	Particle& particle = particles[threadId];

	auto const& cellIdxs = particle.CellIdx;

	int32 const cellX = cellIdxs.x;
	int32 const cellY = cellIdxs.y;
	int32 const cellZ = cellIdxs.z;

	int32 maxNeighbours = 0;

	for(auto const [xOff, yOff, zOff] : lookaround) {
		int32 const xx = cellX + xOff;
		int32 const yy = cellY + yOff;
		int32 const zz = cellZ + zOff;

		if(xx < 0 || yy < 0 || zz < 0 || xx >= config::simulation::boundingBox::xSamples || yy >= config::simulation::boundingBox::ySamples || zz >= config::simulation::boundingBox::zSamples) {
			continue;
		}

		maxNeighbours += grid[zz][yy][xx].Size;
	}

	int32 const blockStart = blockPointer.fetch_add(maxNeighbours);
	
	Particle::Neighbor* neighborBuffer = memoryBuffer + blockStart;
	particle.Neighbors = neighborBuffer;
	int32& neighborCount = particle.NumberOfNeighbors;
	neighborCount = 0;

	int32 particlesTooClose = 0;

	for(auto const [xOff, yOff, zOff] : lookaround) {
		int32 const xx = cellX + xOff;
		int32 const yy = cellY + yOff;
		int32 const zz = cellZ + zOff;

		if(xx < 0 || yy < 0 || zz < 0 || xx >= config::simulation::boundingBox::xSamples || yy >= config::simulation::boundingBox::ySamples || zz >= config::simulation::boundingBox::zSamples) {
			continue;
		}

		Cell& targetCell = grid[zz][yy][xx];
		
		for(int32 i = 0; i < targetCell.Size; ++i) {
			Particle* targetParticle = targetCell.Particles[i];

			auto const direction = targetParticle->Position - particle.Position;

			float const distanceSquared = glm::dot(direction, direction);

			if(distanceSquared <= (config::simulation::physics::smoothingLength * config::simulation::physics::smoothingLength)) {
				if(utils::isCloseToZero(distanceSquared)) {
					++particlesTooClose;
				}
				else {
					neighborBuffer[neighborCount].Distance = std::sqrtf(distanceSquared);
					neighborBuffer[neighborCount].NeighborParticle = targetParticle;

					++neighborCount;
				}
			}
		}
	}

	float density = float(particlesTooClose) * config::simulation::physics::smoothingKernelNormalizationDistanceToDensityConstant * utils::powf(
			utils::powf(config::simulation::physics::smoothingLength, 2), 3);

	for(int32 i = 0; i < neighborCount; ++i) {
		density += config::simulation::physics::smoothingKernelNormalizationDistanceToDensityConstant * utils::powf(
			utils::powf(config::simulation::physics::smoothingLength, 2) - utils::powf(neighborBuffer[i].Distance, 2),
			3
		);
	}

	particle.Density = density;

	particle.Pressure = config::simulation::physics::gasConstantK * (density - config::simulation::physics::restDensity);

}