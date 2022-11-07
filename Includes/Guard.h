#pragma once

#include <type_traits>

template<typename T>
concept Guardable = std::is_member_function_pointer<decltype(&T::Unuse)>::value;

template<Guardable T>
class Guard {
public:
	Guard(T* asset) : Asset(asset) {}

	~Guard() noexcept {
		this->Asset->Unuse();
	}

private:
	T* Asset{};
};