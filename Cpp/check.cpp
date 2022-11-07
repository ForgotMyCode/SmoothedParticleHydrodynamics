#include <check.h>

#include <cassert>
#include <iostream>

template<>
void check<false>(bool condition, std::source_location const location) {
	[[unlikely]]
	if(!condition) {
		std::cerr << "Check failed, at " << location.file_name() << ":" << location.line() << " in " << location.function_name() << "\n";
		assert(false);
		exit(EXIT_FAILURE);
	}
}