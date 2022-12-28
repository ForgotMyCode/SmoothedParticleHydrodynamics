#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR(QUERY) \
	checkCudaError((QUERY), #QUERY, __FILE__, __LINE__)
 
#define CUDA_CHECK_LAST_ERROR() \
	CUDA_CHECK_ERROR(cudaPeekAtLastError())
 
inline
void checkCudaError(cudaError_t error, char const* query, char const* file, int const line) {
	if(error == cudaSuccess) {
		return;
	}
 
	fprintf(stderr, "Cuda ERROR at %s:%d\n\t%s\n\t%s\n", file, line, cudaGetErrorString(error), query);
	assert(false);
	exit(EXIT_FAILURE);
}