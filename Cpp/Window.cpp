#include <Window.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <Scene.h>

#include <iostream>

Window::Window() noexcept {
	Window::ActiveWindow = this;

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	this->WindowHandle = glfwCreateWindow(config::window::defaultWidth, config::window::defaultHeight, config::window::defaultTitle, nullptr, nullptr);
	this->Width = config::window::defaultWidth;
	this->Height = config::window::defaultHeight;
	this->RecalculateProjectionMatrix();
	check(this->WindowHandle);

	glfwMakeContextCurrent(this->WindowHandle);

	check(gladLoadGLLoader((GLADloadproc) glfwGetProcAddress));

	glfwSetFramebufferSizeCallback(this->WindowHandle, &Window::SizeCallback);
	glfwSetKeyCallback(this->WindowHandle, &Window::KeyCallback);
	glfwSetCursorPosCallback(this->WindowHandle, &Window::MouseCallback);

	if constexpr(config::window::hideCursor) {
		glfwSetInputMode(this->WindowHandle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}

	glViewport(GLint(0), GLint(0), GLsizei(this->Width), GLsizei(this->Height));
	glClearColor(config::render::clearColor::r, config::render::clearColor::g, config::render::clearColor::b, 1.f);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_DEPTH_TEST);
	PARANOID_CHECK();

	this->InputManager.RegisterKeyPressedEventListener(GLFW_KEY_ESCAPE, this, &Window::ExitCallback);
	PARANOID_CHECK();

	this->Camera.Initialize();
}

Window::~Window() noexcept {
	glfwTerminate();
}

void Window::Step(float deltaSeconds) {
	glfwPollEvents();

	check(this->ActiveScene);

	this->Camera.Update(deltaSeconds);
	this->ActiveScene->Step(deltaSeconds);
}

void Window::Render() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	PARANOID_CHECK();

	check(this->ActiveScene);

	this->ActiveScene->Render();
	PARANOID_CHECK();

	glfwSwapBuffers(this->WindowHandle);
	PARANOID_CHECK();
}

void Window::SizeCallback(GLFWwindow* windowHandle, int width, int height) {
	PARANOID_CHECK();
	Window* window = Window::GetActiveWindow();

	window->Width = width;
	window->Height = height;

	window->RecalculateProjectionMatrix();
	
	glViewport(GLint(0), GLint(0), GLsizei(window->Width), GLsizei(window->Height));
	PARANOID_CHECK();
}

void Window::KeyCallback(GLFWwindow* windowHandle, int key, int scancode, int action, int mods) {
	PARANOID_CHECK();
	Window* window = Window::GetActiveWindow();

	window->GetInputManager().UpdateMapping(KeyType(key), action != GLFW_RELEASE);
}

void Window::MouseCallback(GLFWwindow* windowHandle, double x, double y) {
	PARANOID_CHECK();

	Window* window = Window::GetActiveWindow();

	window->GetInputManager().UpdateMouse(config::window::mouseSpeed * float(x), config::window::mouseSpeed * float(y));
}

void Window::MainLoop() {
	PARANOID_CHECK();

	float previousFrameTimeSec = 0.f;

	while(!glfwWindowShouldClose(this->WindowHandle)) {
		float const currentFrameTimeSec = float(glfwGetTime());

		float const deltaTimeSec = currentFrameTimeSec - previousFrameTimeSec;
		previousFrameTimeSec = currentFrameTimeSec;

		PARANOID_CHECK();

		this->Step(deltaTimeSec);
		PARANOID_CHECK();

		this->Render();
		PARANOID_CHECK();

		checkOpenGLerror(__FILE__, __LINE__);
	}
}

void Window::SetActiveScene(Scene* scene) {
	PARANOID_CHECK();
	if(this->ActiveScene) {
		this->ActiveScene->Deactivate();
	}

	this->ActiveScene = scene;

	this->ActiveScene->Activate();
	PARANOID_CHECK();
}

void Window::ExitCallback() {
	PARANOID_CHECK();
	glfwSetWindowShouldClose(this->WindowHandle, true);
	PARANOID_CHECK();
}

Window* Window::ActiveWindow = nullptr;