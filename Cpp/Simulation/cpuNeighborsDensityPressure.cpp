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

void Simulation::cpuNeighborsDensityPressure(int32 threadId, Particle* particles, int32 nParticles, Grid grid) {
	if(threadId >= nParticles) {
		return;
	}

	Particle& particle = particles[threadId];

	auto const& cellIdxs = particle.CellIdx;

	int32 const cellX = cellIdxs.x;
	int32 const cellY = cellIdxs.y;
	int32 const cellZ = cellIdxs.z;

	int32& neighborCount = particle.NumberOfNeighbors;
	neighborCount = 0;

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

			float distanceSquared = glm::dot(direction, direction);

			if(distanceSquared <= (config::simulation::physics::smoothingLengthSquared) &&
				targetParticle != &particle
				) {
				particle.Neighbors[neighborCount].Distance = std::sqrtf(distanceSquared);
				particle.Neighbors[neighborCount].NeighborParticle = targetParticle;

				++neighborCount;
			}
		}
	}

	float density = (
		config::simulation::physics::smoothingKernelNormalizationDistanceToDensityConstant * 
		config::simulation::physics::particleMass *
		utils::cube(config::simulation::physics::smoothingLengthSquared)
		);

	for(int32 i = 0; i < neighborCount; ++i) {
		density += (
			config::simulation::physics::smoothingKernelNormalizationDistanceToDensityConstant *
			config::simulation::physics::particleMass *
			utils::cube(config::simulation::physics::smoothingLengthSquared - utils::square(particle.Neighbors[i].Distance)	)
			);
	}

	particle.Density = density;

	particle.Pressure = config::simulation::physics::gasConstantK * (density - config::simulation::physics::restDensity);

}