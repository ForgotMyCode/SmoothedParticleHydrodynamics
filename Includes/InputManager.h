/*****************************************************************//**
 * \file   InputManager.h
 * \brief  Handling user's input and registering listeners.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <functional>
#include <unordered_map>

#include <core.h>

class InputManager {
public:
	using Callback = std::function<void()>;
	using CoordCallback = std::function<void(float, float)>;

	/**
	 * \brief Add new listener of specific key pressed event.
	 * 
	 * \param key Key code to listen for.
	 * \param listener Object to notify.
	 * \param method Object's method to call when the event fires.
	 */
	template<typename Listener>
	void RegisterKeyPressedEventListener(KeyType key, Listener* listener, void(Listener::*method)()) {
		Mapping[key].ReleaseListeners.push_back(std::bind(method, listener));
	}

	/**
	 * \brief Add new listener of specific key released event.
	 *
	 * \param key Key code to listen for.
	 * \param listener Object to notify.
	 * \param method Object's method to call when the event fires.
	 */
	template<typename Listener>
	void RegisterKeyReleasedEventListener(KeyType key, Listener* listener, void(Listener::*method)()) {
		Mapping[key].PressListeners.push_back(std::bind(method, listener));
	}

	/**
	 * \brief Add new listener of mouse move event.
	 * 
	 * \param listener Object to notify.
	 * \param method Object's method to call when the event fires.
	 */
	template<typename Listener>
	void RegisterMouseMoveListener(Listener* listener, void(Listener::*method)(float, float)) {
		MouseMoveListeners.push_back(std::bind_front(method, listener));
	}

	/**
	 * \brief Check if a specific key is currently pressed.
	 * 
	 * \param key Key code.
	 * \return True if the key is currently pressed.
	 */
	bool IsKeyPressed(KeyType key) {
		return Mapping[key].IsSet;
	}

	/**
	 * \brief Should be called by whatever is listening for the actual input. Updates key pressed status and calls relevant listeners.
	 * 
	 * \param key Key code.
	 * \param isKeyPressed Whether or not is the key pressed.
	 */
	void UpdateMapping(KeyType key, bool isKeyPressed) {
		auto keyInfoIt = Mapping.find(key);

		if(keyInfoIt == Mapping.end()) {
			return;
		}

		KeyInfo& info = keyInfoIt->second;

		if(isKeyPressed) {
			if(!info.IsSet) {
				for(auto& pressCallback : info.PressListeners) {
					pressCallback();
				}
			}
		}
		else {
			if(info.IsSet) {
				for(auto& releaseCallback : info.ReleaseListeners) {
					releaseCallback();
				}
			}
		}

		info.IsSet = isKeyPressed;
	}

	/**
	 * \brief Should be called by whatever is listening for the actual input.Calls relevant listeners about mouse status.
	 *
	 * \param x Mouse X.
	 * \param y Mouse Y.
	 */
	void UpdateMouse(float x, float y) {
		for(auto& mouseMoveCallback : MouseMoveListeners) {
			mouseMoveCallback(x, y);
		}
	}

private:
	struct KeyInfo {
		// Whether or not the key is currently pressed.
		bool IsSet{false};

		// Listeners to call when the key is pressed.
		std::vector<Callback> PressListeners;

		// Listeners to call when the key is released.
		std::vector<Callback> ReleaseListeners;
	};

	// Mapping of keys to their info (status & listeners).
	std::unordered_map<KeyType, KeyInfo> Mapping;

	// Registered mouse move listeners.
	std::vector<CoordCallback> MouseMoveListeners;
};