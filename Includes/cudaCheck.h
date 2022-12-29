/*****************************************************************//**
 * \file   cudaCheck.h
 * \brief  Utility functions for handling cuda errors.
 * 
 * \author ondre
 * \date   December 2022
 *********************************************************************/

#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/**
 * \brief Macro wrapper around checkCudaError.
 */
#define CUDA_CHECK_ERROR(QUERY) \
	checkCudaError((QUERY), #QUERY, __FILE__, __LINE__)
 
/**
 * \brief Macro wrapper around cudaPeekAtLastError.
 */
#define CUDA_CHECK_LAST_ERROR() \
	CUDA_CHECK_ERROR(cudaPeekAtLastError())
 
/**
 * \brief Handle CUDA error (crash) if it is not cudaSuccess.
 * 
 * \param error Error to check.
 * \param query Code.
 * \param file File.
 * \param line Line.
 */
inline
void checkCudaError(cudaError_t error, char const* query, char const* file, int const line) {
	if(error == cudaSuccess) {
		return;
	}
 
	fprintf(stderr, "Cuda ERROR at %s:%d\n\t%s\n\t%s\n", file, line, cudaGetErrorString(error), query);
	assert(false);
	exit(EXIT_FAILURE);
}