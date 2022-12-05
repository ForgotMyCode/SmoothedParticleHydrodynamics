#pragma once

#include <type_traits>
#include <utility>

template<typename T>
concept Guardable = std::is_member_function_pointer<decltype(&T::Unuse)>::value;

template<Guardable T>
class Guard {
public:
	Guard(T* asset) : Asset(asset) {}

	Guard& operator=(Guard const&) = delete;
	Guard(Guard const&) = delete;

	Guard& operator=(Guard&& rhs) noexcept {
		this->Asset = std::exchange(rhs.Asset, nullptr);
	}

	Guard(Guard&& rhs) noexcept {
		*this = std::move(rhs);
	}

	~Guard() noexcept {
		if(this->Asset) {
			this->Asset->Unuse();
		}
	}

private:
	T* Asset{};
};