/*****************************************************************//**
 * \file   SceneObject.h
 * \brief  Object that can be in a scene.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

class Scene;

class SceneObject {
public:
	/**
	 * \brief Virtual destructor.
	 * 
	 * \return 
	 */
	virtual ~SceneObject() noexcept = default;

	/**
	 * \brief Called when the scene is initialized.
	 * 
	 */
	virtual void Begin() {}

	/**
	 * \brief Called when the scene ends.
	 * 
	 */
	virtual void End() {}

	/**
	 * \brief Called every frame.
	 * 
	 * \param deltaSeconds Time since the last frame in seconds.
	 */
	virtual void Step([[maybe_unused]] float deltaSeconds) {}

	/**
	 * \brief Render this object.
	 * 
	 */
	virtual void Render() {}

	/**
	 * \brief Set current scene.
	 * 
	 * \param scene New scene. May be null.
	 */
	void SetScene(Scene* scene) {
		this->Scene = scene;
	}

	/**
	 * \brief Get current scene.
	 * 
	 * \return The current scene.
	 */
	Scene* GetScene() {
		return this->Scene;
	}

private:
	// Current scene. May be NULL!
	Scene* Scene{};
};