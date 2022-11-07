#pragma once

#include <vector>

#include <SceneObject.h>

class Scene {
public:

	void AddObject(SceneObject* object) {
		objects.emplace_back(object);
	}

	void Activate() {
		for(auto object : objects) {
			object->SetScene(this);
			object->Begin();
		}
	}

	void Deactivate() {
		for(auto object : objects) {
			object->End();
			object->SetScene(nullptr);
		}
	}

	void Step(float deltaSeconds) {
		for(auto object : objects) {
			object->Step(deltaSeconds);
		}
	}

	void Render() {
		for(auto object : objects) {
			object->Render();
		}
	}

private:
	std::vector<SceneObject*> objects{};
};