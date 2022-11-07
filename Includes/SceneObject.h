#pragma once

class Scene;

class SceneObject {
public:
	virtual ~SceneObject() noexcept = default;

	virtual void Begin() {}

	virtual void End() {}

	virtual void Step(float deltaSeconds) {}

	virtual void Render() {}

	void SetScene(Scene* scene) {
		this->Scene = scene;
	}

	Scene* GetScene() {
		return this->Scene;
	}

private:
	Scene* Scene{};
};