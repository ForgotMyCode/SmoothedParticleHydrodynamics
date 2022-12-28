#pragma once

#include <glm/glm.hpp>
#include <niceIntTypes.h>

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