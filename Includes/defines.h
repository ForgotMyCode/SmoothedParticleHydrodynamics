/*****************************************************************//**
 * \file   defines.h
 * \brief  Header that defines various macros.
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

#pragma once

#define CONSTANT constexpr static

#define CPPGLOBAL extern

#ifndef NDEBUG
#define devCheck(QUERY) \
	if(!(QUERY)) {\
		printf("Check failed! %s:%d  Query: %s\n", __FILE__, __LINE__, #QUERY ); \
		assert(false); \
	}
#else
#define devCheck(QUERY) {}
#endif
 
#define cudaCheckError(QUERY) \
	if((QUERY) != cudaSuccess) { \
		std::cerr << "Cuda ERROR: " << cudaGetErrorString(error) << "\n"; \
		check(false); \
	}
 
#define cudaCheckLastError() \
	cudaCheckError(cudaPeekAtLastError())

using KeyType = unsigned int;

#define SETSUPER(PARENT) \
	using Super = PARENT

void checkOpenGLerror(char const* file, int line);

#define ENABLE_PARANOID_CHECKS

#ifdef ENABLE_PARANOID_CHECKS
#define PARANOID_CHECK() \
	checkOpenGLerror(__FILE__, __LINE__)
#else
#define PARANOID_CHECK() \
	{}
#endif

#define FOR_EACH_CELL_ZYX(ZNAME, YNAME, XNAME) \
	for(int32 (ZNAME) = 0; (ZNAME) < config::simulation::boundingBox::zSamples; ++(ZNAME)) \
			for(int32 (YNAME) = 0; (YNAME) < config::simulation::boundingBox::ySamples; ++(YNAME)) \
				for(int32 (XNAME) = 0; (XNAME) < config::simulation::boundingBox::xSamples; ++(XNAME))

#define FLATTEN_3D_INDEX(X, Y, Z, MAXX, MAXY) \
	(((Z) * (MAXX) * (MAXY)) + ((Y) * (MAXX)) + (X))

#define CUBE(X) \
	((X) * (X) * (X))

#define PI_GPU 3.14159265359f