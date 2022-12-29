/*****************************************************************//**
 * \file   Window.h
 * \brief  Window the user interacts with. It is assumed there is only 1 window.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

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
	/**
	 * \brief Constructor. Initializes everything relevant and show the window.
	 * 
	 * \return 
	 */
	Window() noexcept;

	/**
	 * \brief Destructor.
	 * 
	 * \return 
	 */
	~Window() noexcept;

	/**
	 * \brief Called every frame.
	 * 
	 * \param deltaSeconds Time since the last frame in seconds.
	 */
	void Step(float deltaSeconds);

	/**
	 * \brief Render everything.
	 * 
	 */
	void Render();

	/**
	 * \brief Get window's input manager.
	 * 
	 * \return The input manager
	 */
	InputManager& GetInputManager() {
		return this->InputManager;
	}

	/**
	 * \brief Get window's resource manager.
	 * 
	 * \return 
	 */
	ResourceManager& GetResourceManager() {
		return this->ResourceManager;
	}

	/**
	 * \brief Set active scene.
	 * 
	 * \param scene New scene.
	 */
	void SetActiveScene(Scene* scene);

	/**
	 * \brief Will loop until the exit. Main loop of everything.
	 * 
	 */
	void MainLoop();

	/**
	 * \brief Get currently active window.
	 * 
	 * \return The window.
	 */
	static Window* GetActiveWindow() {
		return Window::ActiveWindow;
	}

	/**
	 * \brief Recalculate projection matrix.
	 * 
	 */
	void RecalculateProjectionMatrix() {
		this->ProjectionMatrix = glm::perspective(
			config::window::fovRadians,
			float(this->Width) / float(this->Height),
			config::window::nearPlaneDistance,
			config::window::farPlaneDistance
		);
	}

	/**
	 * \brief Get cached projection matrix.
	 * 
	 * \return The projection matrix mat4.
	 */
	auto const& GetCachedProjectionMatrix() const {
		return this->ProjectionMatrix;
	}

	/**
	 * \brief Get current camera.
	 * 
	 * \return The camera.
	 */
	Camera& GetCamera() {
		return this->Camera;
	}

	/**
	 * \brief Add the time calculation took to the counter.
	 * 
	 * \param calculationTimeSecs Time the calculation took in seconds.
	 */
	void AddCalculationTime(float calculationTimeSecs) {
		this->CalculationTimeSecs += calculationTimeSecs;
	}

	/**
	 * \brief Get time in seconds. (since GLFW initialization)
	 * 
	 * \return float Time in seconds
	 */
	float GetTimeSeconds() const;

private:
	/**
	 * \brief Callback that handles resized of the window.
	 * 
	 * \param windowHandle Window handle.
	 * \param width New width.
	 * \param height New height.
	 */
	static void SizeCallback(GLFWwindow* windowHandle, int width, int height);

	/**
	 * \brief Callback that handles keyboard events.
	 * 
	 * \param windowHandle Window handle.
	 * \param key Key code.
	 * \param scancode 
	 * \param action Action.
	 * \param mods
	 */
	static void KeyCallback(GLFWwindow* windowHandle, int key, int scancode, int action, int mods);

	/**
	 * \brief Callback that handles mouse events.
	 * 
	 * \param windowHandle Window handle.
	 * \param x Mouse X.
	 * \param y Mouse Y.
	 */
	static void MouseCallback(GLFWwindow* windowHandle, double x, double y);

	/**
	 * \brief Callback that handles exit events.
	 * 
	 */
	void ExitCallback();

	// accumulated calculation time in seconds
	float CalculationTimeSecs{};

	// window resolution
	int64 Width = -1, Height = -1;

	// input manager
	InputManager InputManager{};

	// resource manager
	ResourceManager ResourceManager{};

	// cached projection matrix
	glm::mat4 ProjectionMatrix{};

	// active scene
	Scene* ActiveScene{};

	// camera
	Camera Camera;

	// GLFW window handle
	GLFWwindow* WindowHandle{};

	// Global pointer to active window
	static Window* ActiveWindow;
};