#include <Simulation/gravityUpdate.h>

#include <GLFW/glfw3.h>
#include <iostream>

#include <InputManager.h>
#include <Window.h>

void updateGravity(float deltaSeconds) {
	auto& inputManager = Window::GetActiveWindow()->GetInputManager();
	
	if(inputManager.IsKeyPressed(GLFW_KEY_LEFT)) {
		config::simulation::gravityDirection -= config::simulation::gravityDirectionChangePerSecond * deltaSeconds;

		std::cout << "Gravity direction set to " << config::simulation::gravityDirection << '\n';
	}
	if(inputManager.IsKeyPressed(GLFW_KEY_RIGHT)) {
		config::simulation::gravityDirection += config::simulation::gravityDirectionChangePerSecond * deltaSeconds;

		std::cout << "Gravity direction set to " << config::simulation::gravityDirection << '\n';
	}
}