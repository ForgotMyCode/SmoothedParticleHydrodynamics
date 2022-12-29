/*****************************************************************//**
 * \file   Camera.h
 * \brief  Object in the world that represents player's eye.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>

#include <core.h>

class Camera {
public:

	/**
	 * \brief Constructor. Creates default camera at 0.
	 * 
	 */
	Camera() {
		RecalculateViewMatrix();
	}

	/**
	 * \brief Initialize the camera. Should be called at the start, otherwise the camera will not work properly.
	 * 
	 */
	void Initialize();

	/**
	 * \brief Update step.
	 * 
	 * \param deltaSeconds Delta time in seconds.
	 */
	void Update(float deltaSeconds);

	/**
	 * \brief Get world location of the camera.
	 * 
	 * \return Location vec3.
	 */
	glm::vec3 const& GetLocation() const {
		return this->Location;
	}

	/**
	 * \brief Set world location of the camera.
	 * 
	 * \param newLocation vec3 new world location of the camera.
	 */
	void SetLocation(glm::vec3 const& newLocation) {
		this->Location = newLocation;
	}

	/**
	 * \brief Get direction of the camera.
	 * 
	 * \return vec3 direction of the camera.
	 */
	glm::vec3 const& GetDirection() const {
		return this->Direction;
	}

	/**
	 * \brief Get right direction of the camera.
	 * 
	 * \return vec3 right direction of the camera.
	 */
	glm::vec3 const& GetRight() const {
		return this->Right;
	}

	/**
	 * \brief Set direction of the camera. The given direction is assumed to be normalized.
	 * 
	 * \param newDirection New normalized direction of the camera.
	 */
	void SetNormalizedDirection(glm::vec3 const& newDirection) {
		this->Direction = newDirection;
		this->Right = glm::normalize(glm::cross(newDirection, this->UpVector));

		this->Pitch = glm::asin(this->Direction.y);

		float const invCosPitch = 1.f / glm::cos(this->Pitch);

		this->Yaw = std::atan2f(this->Direction.z * invCosPitch, this->Direction.x * invCosPitch);

		RecalculateViewMatrix();
	}

	/**
	 * \brief Rotates the camera towards the given target.
	 * 
	 * \param target vec3 The target location.
	 */
	void LookAt(glm::vec3 const& target) {
		glm::vec3 direction = glm::normalize(target - Location);
		SetNormalizedDirection(direction);
	}

	/**
	 * \brief Get the precalculated view matrix.
	 * 
	 * \return mat4 view matrix of the camera.
	 */
	auto const& GetCachedViewMatrix() const {
		return this->ViewMatrix;
	}

	/**
	 * \brief Updates the cached view matrix according to currently set params.
	 * 
	 */
	void RecalculateViewMatrix() {
		this->ViewMatrix = glm::lookAt(this->Location, this->Location + this->Direction, this->UpVector);
	}

private:
	/**
	 * \brief Listener of mouse movement events.
	 * 
	 * \param newX new mouse x.
	 * \param newY new mouse y.
	 */
	void MouseMoveListener(float newX, float newY);

	// Up vector
	constexpr static glm::vec3 UpVector{0.f, 1.f, 0.f};

	// World location
	glm::vec3 Location{};

	// Normalized direction
	glm::vec3 Direction{0.f, 0.f, -1.f};

	// Normalized right direction
	glm::vec3 Right{1.f, 0.f, 0.f};

	// Previous mouse x
	float PreviousMouseX{};

	// Previous mouse y
	float PreviousMouseY{};

	// Yaw
	float Yaw{config::window::camera::defaultYaw};

	// Pitch
	float Pitch{};

	// Precalculated view matrix
	glm::mat4 ViewMatrix{};
};