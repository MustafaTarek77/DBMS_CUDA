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
#include "utils.hpp"

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

__global__ void countNonNullFloatsKernel(float* values, int numValues, unsigned int* count);
long long dateTimeToInt64(const char* dateTimeStr);
void int64ToDateTime(long long dateTimeInt, char* dateTimeStr);
__global__ void convertDateTimesToInt64Kernel(char *input_dates, long long *output_dates, int n, int max_str_len, bool findMin);

__global__ void MergeSortGPU(float* keys, int* indices, float* temp_keys, int* temp_indices, long long n, int width, bool isAscending);
__device__ void Merge(float* keys, int* indices, float* temp_keys, int* temp_indices, long long left, long long middle, long long right, bool isAscending);

__device__ void MergeLongLong(long long* keys, int* indices, long long* temp_keys, int* temp_indices, 
                      long long left, long long middle, long long right, bool isAscending);

__global__ void MergeSortGPULongLong(long long* keys, int* indices, long long* temp_keys, int* temp_indices, 
                            long long n, int width, bool isAscending);

__global__ void mergeArraysKernelLongLong(long long* keys_in1, int* indices_in1, int size1,
                                        long long* keys_in2, int* indices_in2, int size2,
                                        long long* keys_out, int* indices_out,
                                        bool isAscending);

__global__ void mergeArraysKernel(float* keys_in1, int* indices_in1, int size1,
                                float* keys_in2, int* indices_in2, int size2,
                                float* keys_out, int* indices_out,
                                bool isAscending);

__global__ void join(void** in1, void** in2, char** names1, char** names2, int* out1, int* out2, Kernel_Condition* flat_conditions, int* group_offsets, int* group_sizes, int conditions_num, int in1_rows_num, int in2_rows_num, int* result_count);
#endif 

