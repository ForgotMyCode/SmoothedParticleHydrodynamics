/*****************************************************************//**
 * \file   cudaargs.h
 * \brief  Small header that fixes some false intellisense errors in visual studio. Also provides a way to pass arguments
 *		when calling cuda kernels instead of <<< >>> .
 * 
 * \author Ondøej Mézl
 * \date   December 2022
 *********************************************************************/

// https://github.com/ForgotMyCode/Visual-Studio-Cuda-IntelliSense-Workaround

#pragma once

#ifdef __INTELLISENSE__
#define CUDA_ARGS(...)
#define CUDA_HIDE_ERRORS(CODE) {}
#ifndef __CUDACC__
#define __CUDACC__
#include <cuda_runtime.h>
#undef __CUDACC__
#else
#include <cuda_runtime.h>
#endif
#else
#define CUDA_ARGS(...) <<< __VA_ARGS__ >>>
#define CUDA_HIDE_ERRORS(CODE) CODE
#endif
