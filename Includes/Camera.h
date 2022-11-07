#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>

#include <core.h>

class Camera {
public:
	Camera() {
		RecalculateViewMatrix();
	}

	void Initialize();

	void Update(float deltaSeconds);

	glm::vec3 const& GetLocation() const {
		return this->Location;
	}

	void SetLocation(glm::vec3 const& newLocation) {
		this->Location = newLocation;
	}

	glm::vec3 const& GetDirection() const {
		return this->Direction;
	}

	glm::vec3 const& GetRight() const {
		return this->Right;
	}

	void SetNormalizedDirection(glm::vec3 const& newDirection) {
		this->Direction = newDirection;
		this->Right = glm::normalize(glm::cross(newDirection, this->UpVector));

		this->Pitch = glm::asin(this->Direction.y);

		float const invCosPitch = 1.f / glm::cos(this->Pitch);

		this->Yaw = std::atan2f(this->Direction.z * invCosPitch, this->Direction.x * invCosPitch);

		RecalculateViewMatrix();
	}

	void LookAt(glm::vec3 const& target) {
		glm::vec3 direction = glm::normalize(target - Location);
		SetNormalizedDirection(direction);
	}

	auto const& GetCachedViewMatrix() const {
		return this->ViewMatrix;
	}

	void RecalculateViewMatrix() {
		this->ViewMatrix = glm::lookAt(this->Location, this->Location + this->Direction, this->UpVector);
	}

private:
	void MouseMoveListener(float newX, float newY);

	constexpr static glm::vec3 UpVector{0.f, 1.f, 0.f};
	glm::vec3 Location{};
	glm::vec3 Direction{0.f, 0.f, -1.f};
	glm::vec3 Right{1.f, 0.f, 0.f};

	float PreviousMouseX{};
	float PreviousMouseY{};

	float Yaw{config::window::camera::defaultYaw};
	float Pitch{};

	glm::mat4 ViewMatrix{};
};