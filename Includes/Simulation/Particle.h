/*****************************************************************//**
 * \file   Particle.h
 * \brief  Particle definition that can be shared between host and device.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <glm/glm.hpp>
#include <niceIntTypes.h>

struct Particle {

	struct Neighbor {

		// Particle being the neighbor
		Particle* NeighborParticle{};

		// how far is the neighbor
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

	// pressure of the particle
	float Pressure{};

	// density of the particle
	float Density{};
};