#pragma once

#include <source_location>

#include <config.h>

template<bool isSkipped = config::debug::noChecks>
void check(bool, std::source_location const = std::source_location::current());

template<>
inline void check<true>(bool, std::source_location const) {}

template<>
void check<false>(bool, std::source_location const);