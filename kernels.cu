#include "kernels.cuh"

// CUDA kernel to find the maximum using warp-level optimizations
__global__ void findWarpMax(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE; // Lane ID within warp
    int warp_id = threadIdx.x / WARP_SIZE; // Warp ID within block
    
    // Each thread loads one element and finds local max
    float local_max = -FLT_MAX;
    if (idx < size) {
        local_max = input[idx];
    }
    
    // Perform warp-level reduction using shuffle operations
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, neighbor);
    }
    
    // First thread in each warp writes result to shared memory
    __shared__ float warp_results[BLOCK_SIZE / WARP_SIZE];
    if (lane_id == 0) {
        warp_results[warp_id] = local_max;
    }
    
    __syncthreads();
    
    // Final warp reduces all partial maxes from all warps
    if (warp_id == 0 && lane_id < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        local_max = warp_results[lane_id];
        
        // Final warp reduction
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            float neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
            local_max = fmaxf(local_max, neighbor);
        }
        
        // First thread in the block writes result to global memory
        if (lane_id == 0) {
            output[blockIdx.x] = local_max;
        }
    }
}

__global__ void findWarpMin(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE; // Lane ID within warp
    int warp_id = threadIdx.x / WARP_SIZE; // Warp ID within block
    
    float local_min = FLT_MAX;  
    if (idx < size) {
        local_min = input[idx];
    }
    
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float neighbor = __shfl_down_sync(0xffffffff, local_min, offset);
        local_min = fminf(local_min, neighbor);  
    }
    
    __shared__ float warp_results[BLOCK_SIZE / WARP_SIZE];
    if (lane_id == 0) {
        warp_results[warp_id] = local_min;
    }
    
    __syncthreads();
    
    if (warp_id == 0 && lane_id < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        local_min = warp_results[lane_id];
        
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            float neighbor = __shfl_down_sync(0xffffffff, local_min, offset);
            local_min = fminf(local_min, neighbor);  
        }
        
        if (lane_id == 0) {
            output[blockIdx.x] = local_min;
        }
    }
}

// Updated kernel for converting dates that's aware of MIN/MAX operation
__global__ void convertDatesToInt64Kernel(char *input_dates, long long *output_dates, int n, int max_str_len, bool findMin) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        char *dateStr = input_dates + idx * max_str_len;
        
        int year = 0, month = 0, day = 0;
        bool parsed = false;
        
        if (dateStr[4] == '-' && dateStr[7] == '-') {
            year = (dateStr[0] - '0') * 1000 + (dateStr[1] - '0') * 100 + (dateStr[2] - '0') * 10 + (dateStr[3] - '0');
            month = (dateStr[5] - '0') * 10 + (dateStr[6] - '0');
            day = (dateStr[8] - '0') * 10 + (dateStr[9] - '0');
        
            if (year > 0 && month >= 1 && month <= 12 && day >= 1 && day <= 31) {
                parsed = true;
            }
        }
        
        if (parsed) {
            output_dates[idx] = (long long)year * 10000 + month * 100 + day;
        } else {
            // Use appropriate sentinel value based on operation
            output_dates[idx] = findMin ? LLONG_MAX : LLONG_MIN;
        }
    }
}

// The helper functions don't need to change:
long long dateToInt64(const char* dateStr) {
    if (dateStr == NULL || dateStr[0] == '\0') {
        return LLONG_MIN;
    }
    
    int year = 0, month = 0, day = 0;
    
    if (sscanf(dateStr, "%d-%d-%d", &year, &month, &day) == 3) {
        if (year <= 0 || month < 1 || month > 12 || day < 1 || day > 31) {
            return LLONG_MIN;  
        }
        
        return (long long)year * 10000 + month * 100 + day;
    } else {
        return LLONG_MIN;
    }
}

void int64ToDate(long long dateInt, char* dateStr) {
    if (dateInt == LLONG_MIN || dateInt == LLONG_MAX) {
        strcpy(dateStr, "Invalid Date");
        return;
    }
    
    if (dateInt == 0) {
        strcpy(dateStr, "0000-00-00");  
        return;
    }
    int year = dateInt / 10000;
    int month = (dateInt % 10000) / 100;
    int day = dateInt % 100;
    
    sprintf(dateStr, "%04d-%02d-%02d", year, month, day);
}
__global__ void findWarpMaxDate(long long* input, long long* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE; // Lane ID within warp
    int warp_id = threadIdx.x / WARP_SIZE; // Warp ID within block
    
    long long local_max = LLONG_MIN;  
    if (idx < size) {
        long long val = input[idx];

        if (val != LLONG_MIN && val > 0) {
            local_max = val;
        }
    }
    
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        long long neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
        if (neighbor != LLONG_MIN && neighbor > 0 && 
            (local_max == LLONG_MIN || neighbor >= local_max)) {
            local_max = neighbor;
        }
    }
    
    __shared__ long long warp_results[BLOCK_SIZE / WARP_SIZE];
    if (lane_id == 0) {
        warp_results[warp_id] = local_max;
    }
    
    __syncthreads();
    
    if (warp_id == 0 && lane_id < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        local_max = warp_results[lane_id];
        
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            long long neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
            local_max = max(local_max, neighbor);
        }
        
        if (lane_id == 0) {
            output[blockIdx.x] = local_max;
        }
    }
}

__global__ void findWarpMinDate(long long* input, long long* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE; // Lane ID within warp
    int warp_id = threadIdx.x / WARP_SIZE; // Warp ID within block
    
    // Each thread loads one element and finds local min
    long long local_min = LLONG_MAX;  // Initialize with invalid date marker
    
    if (idx < size) {
        long long val = input[idx];
        // Only consider valid dates (not LLONG_MAX and not 0)
        if (val != LLONG_MAX && val > 0) {
            local_min = val;
        }
    }
    
    // Perform warp-level reduction using shuffle operations
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        long long neighbor = __shfl_down_sync(0xffffffff, local_min, offset);
        
        // Only update if neighbor has a valid date and is smaller than current min
        if (neighbor != LLONG_MAX && neighbor > 0 && 
            (local_min == LLONG_MAX || neighbor < local_min)) {
            local_min = neighbor;
        }
    }
    
    // First thread in each warp writes result to shared memory
    __shared__ long long warp_results[BLOCK_SIZE / WARP_SIZE];
    if (lane_id == 0) {
        warp_results[warp_id] = local_min;
    }
    
    __syncthreads();
    
    // Final warp reduces all partial mins from all warps
    if (warp_id == 0 && lane_id < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        long long warp_min = warp_results[lane_id];
        
        // For the final reduction, we need to handle the case where some warps found no valid dates
        local_min = (warp_min != LLONG_MAX && warp_min > 0) ? warp_min : LLONG_MAX;
        
        // Final warp reduction
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            long long neighbor = __shfl_down_sync(0xffffffff, local_min, offset);
            
            // Only update if neighbor has a valid date and is smaller than current min
            if (neighbor != LLONG_MAX && neighbor > 0 && 
                (local_min == LLONG_MAX || neighbor < local_min)) {
                local_min = neighbor;
            }
        }
        
        // First thread in the block writes result to global memory
        if (lane_id == 0) {
            output[blockIdx.x] = local_min;
        }
    }
}

// Function to check if a value is null/invalid
__device__ bool isNull(float *val) {
    return *val != *val; // NaN check
}

// Device function for atomic addition of double values
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                          __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Kernel using atomic operations to calculate sum
__global__ void sumKernel(float *data, double *sum, int n) {
    // Each thread handles multiple elements with stride equal to grid size
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    double threadSum = 0.0;
    
    // Process elements with stride pattern to ensure coalesced memory access
    for (int i = idx; i < n; i += stride) {
        if (!isNull(&data[i])) {
            threadSum += (double)data[i];
        }
    }
    
    // Perform block-level reduction first to minimize atomic operations
    __shared__ double blockSum[WARP_SIZE]; // For warp-level reduction
    
    // Initialize shared memory
    if (threadIdx.x < WARP_SIZE) {
        blockSum[threadIdx.x] = 0.0;
    }
    __syncthreads();
    
    // Warp-level reduction first
    unsigned int lane = threadIdx.x % 32;
    unsigned int warpId = threadIdx.x / 32;
    
    // Reduce within each warp using shuffle operations for efficiency
    // #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        blockSum[warpId] = threadSum;
    }
    __syncthreads();
    
    // Final reduction and atomic add by first thread only
    if (threadIdx.x == 0) {
        double finalBlockSum = 0.0;
        for (int i = 0; i < (blockDim.x/WARP_SIZE); i++) {
            finalBlockSum += blockSum[i];
        }
        
        // Atomic add to global sum
        atomicAddDouble(sum, finalBlockSum);
    }
}


__device__ bool isStringNull(const char* str, int maxLen) {
   
    if (str == NULL) return true;
    
    if (str[0] == '\0' || str[0] == ' ') return true;
    
    bool onlySpaces = true;
    for (int i = 0; i < maxLen && str[i] != '\0'; i++) {
        if (str[i] != ' ' && str[i] != '\t' && str[i] != '\r' && str[i] != '\n') {
            onlySpaces = false;
            break;
        }
    }
    
    return onlySpaces;
}

__global__ void countNonNullStringsKernel(char* strings, int numStrings, int maxStringLen, unsigned int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    unsigned int localCount = 0;
    for (int i = idx; i < numStrings; i += stride) {
        char* currentString = strings + i * maxStringLen;
        if (!isStringNull(currentString, maxStringLen)) {
            localCount++;
        }
    }
    
    atomicAdd(count, localCount);
}

__device__ bool isFloatNull(float val) {
    return isnan(val);  // Uses CUDA's built-in isnan function
}

// Kernel to count non-null float values using atomic operations
__global__ void countNonNullFloatsKernel(float* values, int numValues, unsigned int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    unsigned int localCount = 0;
    for (int i = idx; i < numValues; i += stride) {
        if (!isFloatNull(values[i])) {
            localCount++;
        }
    }
    
    // Use atomic add to update the global count
    atomicAdd(count, localCount);
}



