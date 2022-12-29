/*****************************************************************//**
 * \file   Guard.h
 * \brief  Generic guard class that manages access to a resource.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <type_traits>
#include <utility>

/**
 * \brief Requires Unuse() member function.
 */
template<typename T>
concept Guardable = std::is_member_function_pointer<decltype(&T::Unuse)>::value;

template<Guardable T>
class Guard {
public:
	/**
	 * \brief Constructor.
	 * 
	 * \param asset Asset to guard.
	 */
	Guard(T* asset) : Asset(asset) {}

	/**
	 * \brief Guard cannot be copied.
	 * 
	 * \param 
	 * \return 
	 */
	Guard& operator=(Guard const&) = delete;

	/**
	 * \brief Guard cannot be copied.
	 * 
	 * \param 
	 */
	Guard(Guard const&) = delete;

	/**
	 * \brief Move assignment.
	 * 
	 * \param rhs Moved guard.
	 * \return 
	 */
	Guard& operator=(Guard&& rhs) noexcept {
		this->Asset = std::exchange(rhs.Asset, nullptr);
	}

	/**
	 * \brief Move constructor.
	 * 
	 * \param rhs Moved guard.
	 * \return 
	 */
	Guard(Guard&& rhs) noexcept {
		*this = std::move(rhs);
	}

	/**
	 * \brief Dectructor. Calls Unuse() on the guarded resource.
	 * 
	 * \return 
	 */
	~Guard() noexcept {
		if(this->Asset) {
			this->Asset->Unuse();
		}
	}

private:
	// The guarded asset
	T* Asset{};
};