#include <Camera.h>

#include <GLFW/glfw3.h>

#include <core.h>
#include <Window.h>

void Camera::Initialize() {
	Window* window = Window::GetActiveWindow();
	
	window->GetInputManager().RegisterMouseMoveListener(this, &Camera::MouseMoveListener);
}

void Camera::Update(float deltaSeconds) {
	glm::vec3 force{};

	InputManager& inputManager = Window::GetActiveWindow()->GetInputManager();

	float const scale = deltaSeconds * config::window::movementSpeed;

	if(inputManager.IsKeyPressed(GLFW_KEY_W)) {
		force += this->Direction * scale;
	}

	if(inputManager.IsKeyPressed(GLFW_KEY_S)) {
		force -= this->Direction * scale;
	}

	if(inputManager.IsKeyPressed(GLFW_KEY_D)) {
		force += this->Right * scale;
	}

	if(inputManager.IsKeyPressed(GLFW_KEY_A)) {
		force -= this->Right * scale;
	}

	if(inputManager.IsKeyPressed(GLFW_KEY_E)) {
		force += this->UpVector * scale;
	}

	if(inputManager.IsKeyPressed(GLFW_KEY_Q)) {
		force -= this->UpVector * scale;
	}

	this->Location += force;

	RecalculateViewMatrix();
}

void Camera::MouseMoveListener(float x, float y) {
	float const deltaX = x - std::exchange(this->PreviousMouseX, x);
	float const deltaY = y - std::exchange(this->PreviousMouseY, y);

	this->Yaw += deltaX;
	this->Pitch = glm::clamp(this->Pitch - deltaY, config::window::minPitch, config::window::maxPitch);

	const float cosPitch = glm::cos(this->Pitch);

	SetNormalizedDirection(glm::normalize(glm::vec3(
		glm::cos(this->Yaw) * cosPitch,
		glm::sin(this->Pitch),
		glm::sin(this->Yaw) * cosPitch
	)));
}