/*****************************************************************//**
 * \file   Scene.h
 * \brief  Scene that holds and manages various objects of the rendered world.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <vector>

#include <SceneObject.h>

class Scene {
public:
	/**
	 * \brief Add a new object into the scene.
	 * 
	 * \param object Scene object to add.
	 */
	void AddObject(SceneObject* object) {
		objects.emplace_back(object);
	}

	/**
	 * \brief Start the scene. Sets this scene to all of its objects and calls Begin() on them.
	 * 
	 */
	void Activate() {
		for(auto object : objects) {
			object->SetScene(this);
			object->Begin();
		}
	}

	/**
	 * \brief End the scene. Calls End() on all of its objects and sets NULL to their scene.
	 * 
	 */
	void Deactivate() {
		for(auto object : objects) {
			object->End();
			object->SetScene(nullptr);
		}
	}

	/**
	 * \brief Step. Called every frame.
	 * 
	 * \param deltaSeconds Time since the last frame in seconds.
	 */
	void Step(float deltaSeconds) {
		for(auto object : objects) {
			object->Step(deltaSeconds);
		}
	}

	/**
	 * \brief Render this scene. Render all of its objects.
	 * 
	 */
	void Render() {
		for(auto object : objects) {
			object->Render();
		}
	}

private:
	// objects in this scene
	std::vector<SceneObject*> objects{};
};