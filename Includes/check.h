/*****************************************************************//**
 * \file   check.h
 * \brief  Function check for testing conditions that can be easily disabled.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <source_location>

#include <config.h>

/**
 * \brief Templated declaration of check.
 */
template<bool isSkipped = config::debug::noChecks>
void check(bool, std::source_location const = std::source_location::current());

/**
 * \brief Specialization of check when checks should be skipped.
 */
template<>
inline void check<true>(bool, std::source_location const) {}

/**
 * \brief Specialization of check when checks should not be skipped.
 */
template<>
void check<false>(bool, std::source_location const);



