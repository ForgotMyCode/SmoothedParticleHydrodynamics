#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <core.h>
#include <Camera.h>
#include <InputManager.h>
#include <ResourceManager.h>

struct GLFWwindow;
class Scene;

class Window {
public:
	Window() noexcept;

	~Window() noexcept;

	void Step(float deltaSeconds);

	void Render();

	InputManager& GetInputManager() {
		return this->InputManager;
	}

	ResourceManager& GetResourceManager() {
		return this->ResourceManager;
	}

	void SetActiveScene(Scene* scene);

	void MainLoop();

	static Window* GetActiveWindow() {
		return Window::ActiveWindow;
	}

	void RecalculateProjectionMatrix() {
		this->ProjectionMatrix = glm::perspective(
			config::window::fovRadians,
			float(this->Width) / float(this->Height),
			config::window::nearPlaneDistance,
			config::window::farPlaneDistance
		);
	}

	auto const& GetCachedProjectionMatrix() const {
		return this->ProjectionMatrix;
	}

	Camera& GetCamera() {
		return this->Camera;
	}

private:

	static void SizeCallback(GLFWwindow* windowHandle, int width, int height);

	static void KeyCallback(GLFWwindow* windowHandle, int key, int scancode, int action, int mods);

	static void MouseCallback(GLFWwindow* windowHandle, double x, double y);

	void ExitCallback();

	int64 Width = -1, Height = -1;
	InputManager InputManager{};
	ResourceManager ResourceManager{};

	glm::mat4 ProjectionMatrix{};

	Scene* ActiveScene{};

	Camera Camera;

	GLFWwindow* WindowHandle{};

	static Window* ActiveWindow;
};