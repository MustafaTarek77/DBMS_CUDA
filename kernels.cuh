#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <time.h>
#include "config.hpp"

__global__ void findWarpMax(float* input, float* output, int size);
__global__ void findWarpMin(float* input, float* output, int size);


// Function to convert date strings to int64 representation
long long dateToInt64(const char* dateStr);

// Function to convert int64 representation back to date string
void int64ToDate(long long dateInt, char* dateStr);

// CUDA kernels for date operations
__global__ void convertDatesToInt64Kernel(char *input_dates, long long *output_dates, int n, int max_str_len, bool findMin);
__global__ void findWarpMaxDate(long long* input, long long* output, int size);

__global__ void findWarpMinDate(long long* input, long long* output, int size);

__device__ bool isNull(float *val);
__device__ double atomicAddDouble(double* address, double val);
__global__ void sumKernel(float *data, double *sum, int n);

__device__ bool isStringNull(const char* str, int maxLen);
__global__ void countNonNullStringsKernel(char* strings, int numStrings, int maxStringLen, unsigned int* count);

__device__ bool isFloatNull(float val);
__global__ void countNonNullFloatsKernel(float* values, int numValues, unsigned int* count);

#endif 

