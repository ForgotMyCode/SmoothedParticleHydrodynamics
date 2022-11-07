#pragma once

#include <functional>
#include <unordered_map>

#include <core.h>

class InputManager {
public:
	using Callback = std::function<void()>;
	using CoordCallback = std::function<void(float, float)>;

	template<typename Listener>
	void RegisterKeyPressedEventListener(KeyType key, Listener* listener, void(Listener::*method)()) {
		Mapping[key].ReleaseListeners.push_back(std::bind(method, listener));
	}

	template<typename Listener>
	void RegisterKeyReleasedEventListener(KeyType key, Listener* listener, void(Listener::*method)()) {
		Mapping[key].PressListeners.push_back(std::bind(method, listener));
	}

	template<typename Listener>
	void RegisterMouseMoveListener(Listener* listener, void(Listener::*method)(float, float)) {
		MouseMoveListeners.push_back(std::bind_front(method, listener));
	}

	bool IsKeyPressed(KeyType key) {
		return Mapping[key].IsSet;
	}

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

	void UpdateMouse(float x, float y) {
		for(auto& mouseMoveCallback : MouseMoveListeners) {
			mouseMoveCallback(x, y);
		}
	}

private:
	struct KeyInfo {
		bool IsSet{false};
		std::vector<Callback> PressListeners;
		std::vector<Callback> ReleaseListeners;
	};

	std::unordered_map<KeyType, KeyInfo> Mapping;

	std::vector<CoordCallback> MouseMoveListeners;
};