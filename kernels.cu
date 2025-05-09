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

// Kernel to count non-null float values using atomic operations
__global__ void countNonNullFloatsKernel(float* values, int numValues, unsigned int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    unsigned int localCount = 0;
    for (int i = idx; i < numValues; i += stride) {
        if (!isnan(values[i])) {
            localCount++;
        }
    }
    
    // Use atomic add to update the global count
    atomicAdd(count, localCount);
}


long long dateTimeToInt64(const char* dateTimeStr) {
    if (dateTimeStr == nullptr || dateTimeStr[0] == '\0') {
        return LLONG_MIN;
    }
    
    int year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0;
    int items;
    
    // Try format YYYY-MM-DD HH:MM:SS
    items = sscanf(dateTimeStr, "%d-%d-%d %d:%d:%d", &year, &month, &day, &hour, &minute, &second);
    
    // If that didn't work, try YYYY-MM-DD
    if (items < 3) {
        items = sscanf(dateTimeStr, "%d-%d-%d", &year, &month, &day);
        hour = minute = second = 0;
    }
    
    if (items >= 3) {
        if (year <= 0 || month < 1 || month > 12 || day < 1 || day > 31) {
            return LLONG_MIN;  
        }
        
        if (hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 || second > 59) {
            hour = minute = second = 0;
        }
        
        // Store as YYYYMMDDHHMMSS for proper chronological ordering
        return (long long)year * 10000000000LL + 
               (long long)month * 100000000LL + 
               (long long)day * 1000000LL +
               (long long)hour * 10000LL + 
               (long long)minute * 100LL + 
               (long long)second;
    } else {
        return LLONG_MIN;
    }
}

// Convert int64 to date-time string
void int64ToDateTime(long long dateTimeInt, char* dateTimeStr) {
    if (dateTimeInt == LLONG_MIN || dateTimeInt == LLONG_MAX) {
        strcpy(dateTimeStr, "Invalid DateTime");
        return;
    }
    
    if (dateTimeInt == 0) {
        strcpy(dateTimeStr, "0000-00-00 00:00:00");  
        return;
    }
    
    int second = dateTimeInt % 100;
    dateTimeInt /= 100;
    int minute = dateTimeInt % 100;
    dateTimeInt /= 100;
    int hour = dateTimeInt % 100;
    dateTimeInt /= 100;
    int day = dateTimeInt % 100;
    dateTimeInt /= 100;
    int month = dateTimeInt % 100;
    dateTimeInt /= 100;
    int year = dateTimeInt;
    
    sprintf(dateTimeStr, "%04d-%02d-%02d %02d:%02d:%02d", year, month, day, hour, minute, second);
}

// Updated kernel for converting date-times that's aware of MIN/MAX operation
__global__ void convertDateTimesToInt64Kernel(char *input_dates, long long *output_dates, int n, int max_str_len, bool findMin) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        char *dateTimeStr = input_dates + idx * max_str_len;
        
        int year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0;
        bool parsed = false;
        
        // Try to parse YYYY-MM-DD HH:MM:SS format
        if (dateTimeStr[4] == '-' && dateTimeStr[7] == '-' && 
            (dateTimeStr[10] == ' ' || dateTimeStr[10] == 'T') && 
            dateTimeStr[13] == ':' && dateTimeStr[16] == ':') {
            
            year = (dateTimeStr[0] - '0') * 1000 + (dateTimeStr[1] - '0') * 100 + 
                   (dateTimeStr[2] - '0') * 10 + (dateTimeStr[3] - '0');
            month = (dateTimeStr[5] - '0') * 10 + (dateTimeStr[6] - '0');
            day = (dateTimeStr[8] - '0') * 10 + (dateTimeStr[9] - '0');
            hour = (dateTimeStr[11] - '0') * 10 + (dateTimeStr[12] - '0');
            minute = (dateTimeStr[14] - '0') * 10 + (dateTimeStr[15] - '0');
            second = (dateTimeStr[17] - '0') * 10 + (dateTimeStr[18] - '0');
            
            if (year > 0 && month >= 1 && month <= 12 && day >= 1 && day <= 31 &&
                hour >= 0 && hour <= 23 && minute >= 0 && minute <= 59 && second >= 0 && second <= 59) {
                parsed = true;
            }
        }
        // If that failed, try YYYY-MM-DD format
        else if (dateTimeStr[4] == '-' && dateTimeStr[7] == '-') {
            year = (dateTimeStr[0] - '0') * 1000 + (dateTimeStr[1] - '0') * 100 + 
                   (dateTimeStr[2] - '0') * 10 + (dateTimeStr[3] - '0');
            month = (dateTimeStr[5] - '0') * 10 + (dateTimeStr[6] - '0');
            day = (dateTimeStr[8] - '0') * 10 + (dateTimeStr[9] - '0');
            hour = minute = second = 0;
            
            if (year > 0 && month >= 1 && month <= 12 && day >= 1 && day <= 31) {
                parsed = true;
            }
        }
        
        if (parsed) {
            // Store as YYYYMMDDHHMMSS for proper chronological ordering
            output_dates[idx] = (long long)year * 10000000000LL + 
                               (long long)month * 100000000LL + 
                               (long long)day * 1000000LL +
                               (long long)hour * 10000LL + 
                               (long long)minute * 100LL + 
                               (long long)second;
        } else {
            // Use appropriate sentinel value based on operation
            output_dates[idx] = findMin ? LLONG_MAX : LLONG_MIN;
        }
    }
}

// Device function for merging two sorted subarrays
__device__ void Merge(float* keys, int* indices, float* temp_keys, int* temp_indices, 
                      long long left, long long middle, long long right, bool isAscending) {
    long long i = left;
    long long j = middle;
    long long k = left;

    while (i < middle && j < right) {
        bool compareResult;
        if (isAscending)
            compareResult = (keys[i] <= keys[j]);
        else
            compareResult = (keys[i] > keys[j]);
            
        if (compareResult) {
            temp_keys[k] = keys[i];
            temp_indices[k] = indices[i];
            i++;
        } else {
            temp_keys[k] = keys[j];
            temp_indices[k] = indices[j];
            j++;
        }
        k++;
    }

    // Copy remaining elements from left subarray
    while (i < middle) {
        temp_keys[k] = keys[i];
        temp_indices[k] = indices[i];
        i++;
        k++;
    }
    
    // Copy remaining elements from right subarray
    while (j < right) {
        temp_keys[k] = keys[j];
        temp_indices[k] = indices[j];
        j++;
        k++;
    }
}

// GPU Kernel for Merge Sort
__global__ void MergeSortGPU(float* keys, int* indices, float* temp_keys, int* temp_indices, 
                            long long n, int width, bool isAscending) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long left = tid * 2 * width;
    
    // Return early if this thread doesn't have work to do
    if (left >= n)
        return;
        
    long long middle = min(left + width, n);
    long long right = min(left + 2 * width, n);

    // Perform merge operation
    Merge(keys, indices, temp_keys, temp_indices, left, middle, right, isAscending);
}


// Device function for merging two sorted subarrays (long long version)
__device__ void MergeLongLong(long long* keys, int* indices, long long* temp_keys, int* temp_indices, 
                      long long left, long long middle, long long right, bool isAscending) {
    long long i = left;
    long long j = middle;
    long long k = left;

    while (i < middle && j < right) {
        bool compareResult;
        if (isAscending)
            compareResult = (keys[i] <= keys[j]);
        else
            compareResult = (keys[i] > keys[j]);
            
        if (compareResult) {
            temp_keys[k] = keys[i];
            temp_indices[k] = indices[i];
            i++;
        } else {
            temp_keys[k] = keys[j];
            temp_indices[k] = indices[j];
            j++;
        }
        k++;
    }

    // Copy remaining elements from left subarray
    while (i < middle) {
        temp_keys[k] = keys[i];
        temp_indices[k] = indices[i];
        i++;
        k++;
    }
    
    // Copy remaining elements from right subarray
    while (j < right) {
        temp_keys[k] = keys[j];
        temp_indices[k] = indices[j];
        j++;
        k++;
    }
}

__global__ void MergeSortGPULongLong(long long* keys, int* indices, long long* temp_keys, int* temp_indices, 
                            long long n, int width, bool isAscending) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long left = tid * 2 * (long long)width;
    
    // Return early if this thread doesn't have work to do
    if (left >= n)
        return;
        
    long long middle = min(left + width, n);
    long long right = min(left + 2 * (long long)width, n);

    // Perform merge operation
    MergeLongLong(keys, indices, temp_keys, temp_indices, left, middle, right, isAscending);
}

// Kernel to merge two sorted arrays on the GPU (float version)
__global__ void mergeArraysKernel(float* keys_in1, int* indices_in1, int size1,
                                float* keys_in2, int* indices_in2, int size2,
                                float* keys_out, int* indices_out,
                                bool isAscending) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= size1 + size2) return;
    
    // Binary search to find the position of this thread's element
    int pos;
    if (tid < size1) {
        // This thread handles an element from the first array
        float key = keys_in1[tid];
        int idx = indices_in1[tid];
        
        // Find where this element should go in the merged array
        int low = 0;
        int high = size2;
        
        while (low < high) {
            int mid = (low + high) / 2;
            bool pred;
            if (isAscending) {
                pred = key <= keys_in2[mid];
            } else {
                pred = key >= keys_in2[mid];
            }
            
            if (pred) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        
        pos = tid + low;
        
        // Write to output array
        keys_out[pos] = key;
        indices_out[pos] = idx;
    } else {
        // This thread handles an element from the second array
        int tid2 = tid - size1;
        float key = keys_in2[tid2];
        int idx = indices_in2[tid2];
        
        // Find where this element should go in the merged array
        int low = 0;
        int high = size1;
        
        while (low < high) {
            int mid = (low + high) / 2;
            bool pred;
            if (isAscending) {
                pred = keys_in1[mid] <= key;
            } else {
                pred = keys_in1[mid] >= key;
            }
            
            if (pred) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        
        pos = low + tid2;
        
        // Write to output array
        keys_out[pos] = key;
        indices_out[pos] = idx;
    }
}

// Kernel to merge two sorted arrays on the GPU (long long version)
__global__ void mergeArraysKernelLongLong(long long* keys_in1, int* indices_in1, int size1,
                                        long long* keys_in2, int* indices_in2, int size2,
                                        long long* keys_out, int* indices_out,
                                        bool isAscending) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= size1 + size2) return;
    
    // Binary search to find the position of this thread's element
    int pos;
    if (tid < size1) {
        // This thread handles an element from the first array
        long long key = keys_in1[tid];
        int idx = indices_in1[tid];
        
        // Find where this element should go in the merged array
        int low = 0;
        int high = size2;
        
        while (low < high) {
            int mid = (low + high) / 2;
            bool pred;
            if (isAscending) {
                pred = key <= keys_in2[mid];
            } else {
                pred = key >= keys_in2[mid];
            }
            
            if (pred) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        
        pos = tid + low;
        
        // Write to output array
        keys_out[pos] = key;
        indices_out[pos] = idx;
    } else {
        // This thread handles an element from the second array
        int tid2 = tid - size1;
        long long key = keys_in2[tid2];
        int idx = indices_in2[tid2];
        
        // Find where this element should go in the merged array
        int low = 0;
        int high = size1;
        
        while (low < high) {
            int mid = (low + high) / 2;
            bool pred;
            if (isAscending) {
                pred = keys_in1[mid] <= key;
            } else {
                pred = keys_in1[mid] >= key;
            }
            
            if (pred) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        
        pos = low + tid2;
        
        // Write to output array
        keys_out[pos] = key;
        indices_out[pos] = idx;
    }
}

__device__ bool compareFloats(float lhs, const char& op, float rhs) {
    if (op == '=') return lhs == rhs;
    if (op == '!') return lhs != rhs;
    if (op == '<') return lhs < rhs;
    if (op == '>') return lhs > rhs;
    return false;
}

__device__ bool compareStrings(const char* lhs, const char& op, const char* rhs) {
    if (op == '=') return lhs == rhs;
    if (op == '!') return lhs != rhs;
    if (op == '<') return lhs < rhs;
    if (op == '>') return lhs > rhs;
    return false;
}

// __device__ int parseDateTime(const char* datetime) {
//     // Format: "YYYY-MM-DD HH:MM:SS"
//     int year = 0, month = 0, day = 0, hour = 0, min = 0, sec = 0;

//     year  = (datetime[0]-'0')*1000 + (datetime[1]-'0')*100 + (datetime[2]-'0')*10 + (datetime[3]-'0');
//     month = (datetime[5]-'0')*10 + (datetime[6]-'0');
//     day   = (datetime[8]-'0')*10 + (datetime[9]-'0');
//     hour  = (datetime[11]-'0')*10 + (datetime[12]-'0');
//     min   = (datetime[14]-'0')*10 + (datetime[15]-'0');
//     sec   = (datetime[17]-'0')*10 + (datetime[18]-'0');
    
//     // Simple conversion to seconds since a base date 
//     return (((year * 12 + month) * 31 + day) * 24 + hour) * 60 * 60 + min * 60 + sec;
// }

__device__ int parseDateTime(const char* datetime) {
    // Handle optional ::TIMESTAMP suffix
    int len = 0;
    while (datetime[len] != '\0') ++len;

    // Find start of actual timestamp
    const char* ptr = datetime;

    // If the string is quoted, skip the quotes
    if (*ptr == '\'') ++ptr;

    // Check if the datetime ends with '::TIMESTAMP'
    const char* ts_suffix = "::TIMESTAMP";
    int suffix_len = 11;
    if (len >= suffix_len) {
        bool has_suffix = true;
        for (int i = 0; i < suffix_len; ++i) {
            if (datetime[len - suffix_len + i] != ts_suffix[i]) {
                has_suffix = false;
                break;
            }
        }
        if (has_suffix) {
            // Make a null-terminated copy without the suffix for parsing
            char clean[20];
            for (int i = 0; i < 19 && ptr[i] != '\0'; ++i) {
                clean[i] = ptr[i];
                clean[i + 1] = '\0';
            }
            datetime = clean;
        }
    }

    // Parse cleaned format: "YYYY-MM-DD HH:MM:SS"
    int year = 0, month = 0, day = 0, hour = 0, min = 0, sec = 0;

    year  = (datetime[0]-'0')*1000 + (datetime[1]-'0')*100 + (datetime[2]-'0')*10 + (datetime[3]-'0');
    month = (datetime[5]-'0')*10 + (datetime[6]-'0');
    day   = (datetime[8]-'0')*10 + (datetime[9]-'0');
    hour  = (datetime[11]-'0')*10 + (datetime[12]-'0');
    min   = (datetime[14]-'0')*10 + (datetime[15]-'0');
    sec   = (datetime[17]-'0')*10 + (datetime[18]-'0');

    // Simple conversion to seconds since a base date 
    return (((year * 12 + month) * 31 + day) * 24 + hour) * 60 * 60 + min * 60 + sec;
}


__device__ bool compareDates(const char* lhs, const char& op, const char* rhs) {
    int lhs_time = parseDateTime(lhs);
    int rhs_time = parseDateTime(rhs);

    if (op == '=') return lhs_time == rhs_time;
    if (op == '!') return lhs_time != rhs_time;
    if (op == '<') return lhs_time < rhs_time;
    if (op == '>') return lhs_time > rhs_time;

    return false; // Unsupported op
}

__global__ void join(void** in1, void** in2, char** names1, char** names2, int* out1, int* out2, Kernel_Condition* flat_conditions, int* group_offsets, int* group_sizes, int conditions_num, int in1_rows_num, int in2_rows_num, int* result_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = in1_rows_num * in2_rows_num;

    if (idx < total) {
        int row1 = idx / in2_rows_num;
        int row2 = idx % in2_rows_num;

        bool match_found = false;
        if(conditions_num>0) {
            for (int g = 0; g < conditions_num && !match_found; ++g) {
                int group_start = group_offsets[g];
                int group_size = group_sizes[g];
                bool group_ok = true;

                for (int c = 0; c < group_size && group_ok; ++c) {
                    Kernel_Condition cond = flat_conditions[group_start + c];

                    // printf("Condition: %d, idx1: %d, idx2: %d, type: %c, op: %c,\n",
                    //        c, cond.idx1, cond.idx2, cond.type, cond.relational_operator);

                    if (cond.type == 'N') {
                        float val1 = ((float*)in1[cond.idx1])[row1];
                        float val2 = 0;
                        if (cond.idx2==-1) {
                            val2 = *((float*)cond.value);
                        }
                        else {
                            val2 = ((float*)in2[cond.idx2])[row2];
                        }
                        if (!compareFloats(val1, cond.relational_operator, val2)) {
                            group_ok = false;
                        }
                    }
                    else if (cond.type == 'T') {
                        char* val1 = ((char**)in1[cond.idx1])[row1];
                        char* val2;
                        if (cond.idx2==-1) {
                            val2 = ((char*)cond.value);
                        }
                        else {
                            val2 = ((char**)in2[cond.idx2])[row2];
                        }
                        if (!compareStrings(val1, cond.relational_operator, val2)) {
                            group_ok = false;
                        }
                    }
                    else if (cond.type == 'D') {
                        char* val1 = ((char**)in1[cond.idx1])[row1];
                        char* val2;
                        if (cond.idx2==-1) {
                            val2 = ((char*)cond.value);
                        }
                        else {
                            val2 = ((char**)in2[cond.idx2])[row2];
                        }
                        if (!compareDates(val1, cond.relational_operator, val2)) {
                            group_ok = false;
                        }
                    }
                }

                if (group_ok) {
                    match_found = true;
                }
            }

            if (match_found) {
                int pos = atomicAdd(result_count, 1);
                out1[pos] = row1;
                out2[pos] = row2;
            }
        }
        else {
            int pos = atomicAdd(result_count, 1);
            out1[pos] = row1;
            out2[pos] = row2;
        }
    }
}
