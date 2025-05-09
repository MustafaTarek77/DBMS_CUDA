#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <time.h>
#include "config.hpp"
#include "kernels.cuh"
#include "db_utils.hpp"
#include "utils.hpp"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include "db_utils_cpu.hpp"

// Function to remove outermost balanced parentheses
std::string removeOuterParentheses(const std::string& expr) {
    std::string result = trim(expr);
    
    while (result.front() == '(' && result.back() == ')') {
        // Check if these are balanced outer parentheses
        int level = 0;
        bool balanced = true;
        
        for (size_t i = 0; i < result.length() - 1; i++) {
            if (result[i] == '(') level++;
            else if (result[i] == ')') level--;
            
            if (level == 0 && i < result.length() - 1) {
                balanced = false;
                break;
            }
        }
        
        if (balanced) {
            result = result.substr(1, result.length() - 2);
            result = trim(result);
        } else {
            break;
        }
    }
    
    return result;
}

// Function to split by word boundaries to handle AND/OR operators
std::vector<std::string> splitByWord(const std::string& str, const std::string& word) {
    std::vector<std::string> result;
    std::string temp = str;
    
    // Create a regex pattern that matches the word as a whole word
    std::regex pattern("\\b" + word + "\\b");
    
    // Split the string by the word
    std::sregex_token_iterator iter(temp.begin(), temp.end(), pattern, -1);
    std::sregex_token_iterator end;
    
    while (iter != end) {
        // Add each part that's not empty after trimming
        std::string part = trim(*iter);
        if (!part.empty()) {
            result.push_back(part);
        }
        ++iter;
    }
    
    return result;
}

// Function to split a string by an operator, respecting parentheses
std::vector<std::string> splitByOperator(const std::string& str, const std::string& op) {
    std::vector<std::string> result;
    int parenthesis_level = 0;
    size_t last_pos = 0;
    
    for (size_t i = 0; i < str.length(); i++) {
        if (str[i] == '(') {
            parenthesis_level++;
        } else if (str[i] == ')') {
            parenthesis_level--;
        } else if (parenthesis_level == 0) {
            // Check if we've found the operator
            if (i + op.length() <= str.length() && 
                str.substr(i, op.length()) == op && 
                (i == 0 || !isalnum(str[i-1])) && 
                (i + op.length() == str.length() || !isalnum(str[i+op.length()]))) {
                
                // Add the part before the operator
                std::string part = trim(str.substr(last_pos, i - last_pos));
                if (!part.empty()) {
                    result.push_back(part);
                }
                
                // Move past the operator
                i += op.length() - 1; // -1 because loop will increment i
                last_pos = i + 1;
            }
        }
    }
    
    // Add the last part
    std::string part = trim(str.substr(last_pos));
    if (!part.empty()) {
        result.push_back(part);
    }
    
    return result;
}

// Function to parse complex expression with AND and OR operators
void parseComplexExpression(const std::string& expr, std::vector<std::vector<Condition>>& conditions) {   
    // Remove outer parentheses if present
    std::string clean_expr = removeOuterParentheses(expr);
    
    // Split the expression by OR at the top level
    std::vector<std::string> or_parts;
    int parenthesis_level = 0;
    size_t start_pos = 0;
    
    for (size_t i = 0; i < clean_expr.length(); i++) {
        if (clean_expr[i] == '(') {
            parenthesis_level++;
        } else if (clean_expr[i] == ')') {
            parenthesis_level--;
        } else if (parenthesis_level == 0 && 
                  i + 2 < clean_expr.length() && 
                  clean_expr.substr(i, 2) == "OR" && 
                  (i == 0 || !isalnum(clean_expr[i-1])) && 
                  (i+2 >= clean_expr.length() || !isalnum(clean_expr[i+2]))) {
            
            // Add the part before the OR
            std::string part = trim(clean_expr.substr(start_pos, i - start_pos));
            if (!part.empty()) {
                or_parts.push_back(part);
            }
            
            // Move past the OR
            i += 1; // Skip "OR" (loop will increment i)
            start_pos = i + 1;
        }
    }
    
    // Add the last part
    std::string last_part = trim(clean_expr.substr(start_pos));
    if (!last_part.empty()) {
        or_parts.push_back(last_part);
    }
    
    // std::cout << "Split by OR (" << or_parts.size() << " parts):" << std::endl;
    // for (size_t i = 0; i < or_parts.size(); i++) {
    //     std::cout << i + 1 << ": \"" << or_parts[i] << "\"" << std::endl;
    // }
    
    // Process each OR part
    for (const auto& or_part : or_parts) {
        std::vector<Condition> and_conditions;
        
        // Remove outer parentheses again
        std::string clean_or_part = removeOuterParentheses(or_part);
        
        // Split by AND at the top level
        std::vector<std::string> and_parts;
        parenthesis_level = 0;
        start_pos = 0;
        
        for (size_t i = 0; i < clean_or_part.length(); i++) {
            if (clean_or_part[i] == '(') {
                parenthesis_level++;
            } else if (clean_or_part[i] == ')') {
                parenthesis_level--;
            } else if (parenthesis_level == 0 && 
                      i + 3 < clean_or_part.length() && 
                      clean_or_part.substr(i, 3) == "AND" && 
                      (i == 0 || !isalnum(clean_or_part[i-1])) && 
                      (i+3 >= clean_or_part.length() || !isalnum(clean_or_part[i+3]))) {
                
                // Add the part before the AND
                std::string part = trim(clean_or_part.substr(start_pos, i - start_pos));
                if (!part.empty()) {
                    and_parts.push_back(part);
                }
                
                // Move past the AND
                i += 2; // Skip "AND" (loop will increment i)
                start_pos = i + 1;
            }
        }
        
        // Add the last part
        std::string last_and_part = trim(clean_or_part.substr(start_pos));
        if (!last_and_part.empty()) {
            and_parts.push_back(last_and_part);
        }
        
        // std::cout << "  Split by AND (" << and_parts.size() << " parts):" << std::endl;
        // for (size_t i = 0; i < and_parts.size(); i++) {
        //     std::cout << "    " << i + 1 << ": \"" << and_parts[i] << "\"" << std::endl;
        // }
        
        // Parse each simple condition
        for (const auto& and_part : and_parts) {
            std::string clean_and_part = removeOuterParentheses(and_part);
            
            // Parse simple condition
            std::regex condition_pattern(R"((\S+)\s*(=|!=|<|>|>=|<=)\s*('(?:[^']|'')*'(?:\s*::\s*\w+)?|\S+))");

            std::smatch match;
            
            if (std::regex_match(clean_and_part, match, condition_pattern)) {
                Condition condition;
                condition.left_operand = match[1];
                condition.relational_operator = match[2];
                condition.right_operand = match[3];
                
                and_conditions.push_back(condition);
                
                // std::cout << "    Parsed condition: " << condition.left_operand 
                //           << " " << condition.relational_operator 
                //           << " " << condition.right_operand << std::endl;
            } else {
                std::cerr << "    Failed to parse condition: " << clean_and_part << std::endl;
            }
        }
        
        // Add this group of AND conditions to the result
        if (!and_conditions.empty()) {
            conditions.push_back(and_conditions);
        }
    }
}

// Function to print the parsed condition structure
void printConditionStructure(const std::vector<std::vector<Condition>>& conditions) {
    std::cout << "\nParsed condition structure:" << std::endl;
    for (size_t i = 0; i < conditions.size(); i++) {
        std::cout << "OR group " << i + 1 << ":" << std::endl;
        for (const auto& condition : conditions[i]) {
            std::cout << "  " << condition.left_operand << " " 
                      << condition.relational_operator << " " 
                      << condition.right_operand << std::endl;
        }
    }
}


void WriteAggregationToCSV(
    const std::string& function, 
    const std::string& columnName, 
    const void* result, 
    const std::string& columnType,
    const std::string& resultValueType = "FLOAT",
    const std::string& customFilename = "") 
{
    // Generate filename if not provided
    std::string filename;
    if (customFilename.empty()) {
        // Generate a timestamp for the CSV filename
        time_t now = time(0);
        struct tm timeinfo;
        char timestamp[20];
        #ifdef _WIN32
        localtime_s(&timeinfo, &now);
        #else
        localtime_r(&now, &timeinfo);
        #endif
        strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &timeinfo);
        
        // Clean the column name for the filename (remove special characters)
        std::string cleanColumnName = columnName;
        size_t typePos = cleanColumnName.find('(');
        if (typePos != std::string::npos) {
            cleanColumnName = cleanColumnName.substr(0, typePos);
        }
        
        // Remove any characters that might not be suitable for filenames
        cleanColumnName.erase(
            std::remove_if(cleanColumnName.begin(), cleanColumnName.end(), 
                          [](char c) { return !std::isalnum(c); }),
            cleanColumnName.end());
            
        filename = function + "_" + cleanColumnName + "_" + timestamp + ".csv";
    } else {
        filename = customFilename;
    }
    
    // Create and open CSV file
    std::ofstream csvFile(filename);
    if (!csvFile.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write header: function(columnName)
    csvFile << function << "(" << columnName << ")" << std::endl;
    
    // Write the result based on data type and result value type
    if (resultValueType == "UINT") {
        // For unsigned integer results (like count)
        unsigned int uintResult = *static_cast<const unsigned int*>(result);
        csvFile << uintResult << std::endl;
    }
    else if (columnType == "FLOAT" || columnType == "TEXT" || columnType == "DATE") {
        if (resultValueType == "DOUBLE") {
            double doubleResult = *static_cast<const double*>(result);
            csvFile << std::fixed << std::setprecision(6) << doubleResult << std::endl;
        } else {
            float floatResult = *static_cast<const float*>(result);
            csvFile << std::fixed << std::setprecision(6) << floatResult << std::endl;
        }
    } 
    else {
        // Generic fallback - just convert to string if type is unknown
        if (resultValueType == "DOUBLE") {
            double doubleResult = *static_cast<const double*>(result);
            csvFile << std::fixed << std::setprecision(6) << doubleResult << std::endl;
        } else {
            float floatResult = *static_cast<const float*>(result);
            csvFile << std::fixed << std::setprecision(6) << floatResult << std::endl;
        }
    }
    
    csvFile.close();
    //std::cout << "Aggregation result saved to " << filename << std::endl;
}


void mergeSortGPU(float* d_keys, int* d_indices, long long n, bool isAscending, cudaStream_t stream = 0) {
    // Allocate temporary arrays on device
    float* d_temp_keys = NULL;
    int* d_temp_indices = NULL;
    
    cudaError_t err;
    err = cudaMalloc((void**)&d_temp_keys, n * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in allocating d_temp_keys: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc((void**)&d_temp_indices, n * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in allocating d_temp_indices: %s\n", cudaGetErrorString(err));
        cudaFree(d_temp_keys);
        return;
    }
    
    // Perform merge sort in a bottom-up manner
    float* keys = d_keys;
    int* indices = d_indices;
    float* temp_keys = d_temp_keys;
    int* temp_indices = d_temp_indices;
    
    // Bottom-up merge sort
    for (int width = 1; width < n; width *= 2) {
        // Calculate grid size
        int numThreadsNeeded = (n + 2 * width - 1) / (2 * width);
        int numBlocks = (numThreadsNeeded + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Launch kernel on specified stream
        MergeSortGPU<<<numBlocks, BLOCK_SIZE, 0, stream>>>(keys, indices, temp_keys, temp_indices, n, width, isAscending);
        
        // Check for errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error in MergeSortGPU (width=%d): %s\n", 
                   width, cudaGetErrorString(err));
            cudaFree(d_temp_keys);
            cudaFree(d_temp_indices);
            return;
        }
        
        // Swap pointers for next iteration
        float* tmp_keys = keys;
        int* tmp_indices = indices;
        keys = temp_keys;
        indices = temp_indices;
        temp_keys = tmp_keys;
        temp_indices = tmp_indices;
    }
    
    // If the final result is in temporary arrays, copy back to original arrays
    if (keys != d_keys) {
        cudaMemcpyAsync(d_keys, keys, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_indices, indices, n * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    }
    
    // Create event to track completion
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream);
    cudaEventSynchronize(event);
    cudaEventDestroy(event);
    
    cudaFree(d_temp_keys);
    cudaFree(d_temp_indices);
}

void mergeSortGPULongLong(long long* d_keys, int* d_indices, long long n, bool isAscending, cudaStream_t stream = 0) {
    // Allocate temporary arrays on device
    long long* d_temp_keys = NULL;
    int* d_temp_indices = NULL;
    
    cudaError_t err;
    err = cudaMalloc((void**)&d_temp_keys, n * sizeof(long long));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in allocating d_temp_keys: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc((void**)&d_temp_indices, n * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in allocating d_temp_indices: %s\n", cudaGetErrorString(err));
        cudaFree(d_temp_keys);
        return;
    }
    
    // Perform merge sort in a bottom-up manner
    long long* keys = d_keys;
    int* indices = d_indices;
    long long* temp_keys = d_temp_keys;
    int* temp_indices = d_temp_indices;
    
    // Maximum width to avoid integer overflow
    const int MAX_WIDTH = 1073741824; // 2^30
    
    // Bottom-up merge sort
    for (int width = 1; width < n && width < MAX_WIDTH; width *= 2) {
        // Calculate grid size
        int numThreadsNeeded = (n + 2 * width - 1) / (2 * width);
        int numBlocks = (numThreadsNeeded + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        if (numBlocks > 0) { // Ensure we launch at least one block
            // Launch kernel on specified stream
            MergeSortGPULongLong<<<numBlocks, BLOCK_SIZE, 0, stream>>>(keys, indices, temp_keys, temp_indices, n, width, isAscending);
            
            // Check for errors
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error in MergeSortGPULongLong (width=%d): %s\n", 
                       width, cudaGetErrorString(err));
                cudaFree(d_temp_keys);
                cudaFree(d_temp_indices);
                return;
            }
            
            // Swap pointers for next iteration
            long long* tmp_keys = keys;
            int* tmp_indices = indices;
            keys = temp_keys;
            indices = temp_indices;
            temp_keys = tmp_keys;
            temp_indices = tmp_indices;
        }
    }
    
    // If the final result is in temporary arrays, copy back to original arrays
    if (keys != d_keys) {
        cudaMemcpyAsync(d_keys, keys, n * sizeof(long long), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_indices, indices, n * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    }
    
    // Create event to track completion
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream);
    cudaEventSynchronize(event);
    cudaEventDestroy(event);
    
    cudaFree(d_temp_keys);
    cudaFree(d_temp_indices);
}

void mergeSortedArrays(float* keys, int* indices, float* tempKeys, int* tempIndices, 
                       int start, int mid, int end, bool isAscending) {
    int i = start;
    int j = mid;
    int k = start;
    
    while (i < mid && j < end) {
        bool compareResult;
        if (isAscending) {
            compareResult = keys[i] <= keys[j];
        } else {
            compareResult = keys[i] >= keys[j];
        }
        
        if (compareResult) {
            tempKeys[k] = keys[i];
            tempIndices[k] = indices[i];
            i++;
        } else {
            tempKeys[k] = keys[j];
            tempIndices[k] = indices[j];
            j++;
        }
        k++;
    }
    
    // Copy remaining elements
    while (i < mid) {
        tempKeys[k] = keys[i];
        tempIndices[k] = indices[i];
        i++;
        k++;
    }
    
    while (j < end) {
        tempKeys[k] = keys[j];
        tempIndices[k] = indices[j];
        j++;
        k++;
    }
    
    // Copy back to original array
    for (i = start; i < end; i++) {
        keys[i] = tempKeys[i];
        indices[i] = tempIndices[i];
    }
}

void mergeSortedArraysLongLong(long long* keys, int* indices, long long* tempKeys, int* tempIndices, 
                              int start, int mid, int end, bool isAscending) {
    int i = start;
    int j = mid;
    int k = start;
    
    while (i < mid && j < end) {
        bool compareResult;
        if (isAscending) {
            compareResult = keys[i] <= keys[j];
        } else {
            compareResult = keys[i] >= keys[j];
        }
        
        if (compareResult) {
            tempKeys[k] = keys[i];
            tempIndices[k] = indices[i];
            i++;
        } else {
            tempKeys[k] = keys[j];
            tempIndices[k] = indices[j];
            j++;
        }
        k++;
    }
    
    // Copy remaining elements
    while (i < mid) {
        tempKeys[k] = keys[i];
        tempIndices[k] = indices[i];
        i++;
        k++;
    }
    
    while (j < end) {
        tempKeys[k] = keys[j];
        tempIndices[k] = indices[j];
        j++;
        k++;
    }
    
    // Copy back to original array
    for (i = start; i < end; i++) {
        keys[i] = tempKeys[i];
        indices[i] = tempIndices[i];
    }
}

// Merge all sorted chunks for float data
void mergeChunks(float* keys, int* indices, long long numRows, int numChunks, long long chunkSize, bool isAscending) {
    // Allocate temporary arrays for merging
    float* tempKeys = new float[numRows];
    int* tempIndices = new int[numRows];
    
    // Start with chunkSize and double until we cover the whole array
    for (long long currentSize = chunkSize; currentSize < numRows; currentSize *= 2) {
        for (long long start = 0; start < numRows; start += 2 * currentSize) {
            long long mid = std::min(start + currentSize, numRows);
            long long end = std::min(start + 2 * currentSize, numRows);
            
            if (mid < end) {
                mergeSortedArrays(keys, indices, tempKeys, tempIndices, start, mid, end, isAscending);
            }
        }
    }
    
    delete[] tempKeys;
    delete[] tempIndices;
}

// Merge all sorted chunks for long long data
void mergeChunksLongLong(long long* keys, int* indices, long long numRows, int numChunks, long long chunkSize, bool isAscending) {
    // Allocate temporary arrays for merging
    long long* tempKeys = new long long[numRows];
    int* tempIndices = new int[numRows];
    
    // Start with chunkSize and double until we cover the whole array
    for (long long currentSize = chunkSize; currentSize < numRows; currentSize *= 2) {
        for (long long start = 0; start < numRows; start += 2 * currentSize) {
            long long mid = std::min(start + currentSize, numRows);
            long long end = std::min(start + 2 * currentSize, numRows);
            
            if (mid < end) {
                mergeSortedArraysLongLong(keys, indices, tempKeys, tempIndices, start, mid, end, isAscending);
            }
        }
    }
    
    delete[] tempKeys;
    delete[] tempIndices;
}

void mergeChunksGPU(float* h_keys, int* h_indices, long long numRows, int numChunks, long long chunkSize, bool isAscending) {
    // We can do the full merge in one go
    // Allocate device memory for input
    float* d_keys = nullptr;
    int* d_indices = nullptr;
    cudaMalloc(&d_keys, numRows * sizeof(float));
    cudaMalloc(&d_indices, numRows * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_keys, h_keys, numRows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, numRows * sizeof(int), cudaMemcpyHostToDevice);
    
    // Allocate temporary device memory for output
    float* d_temp_keys = nullptr;
    int* d_temp_indices = nullptr;
    cudaMalloc(&d_temp_keys, numRows * sizeof(float));
    cudaMalloc(&d_temp_indices, numRows * sizeof(int));
    
    // Merge chunks in parallel
    float* inputKeys = d_keys;
    int* inputIndices = d_indices;
    float* outputKeys = d_temp_keys;
    int* outputIndices = d_temp_indices;
    
    for (long long currentSize = chunkSize; currentSize < numRows; currentSize *= 2) {
        // For each pair of chunks at current level
        for (long long start = 0; start < numRows; start += 2 * currentSize) {
            long long mid = std::min(start + currentSize, numRows);
            long long end = std::min(start + 2 * currentSize, numRows);
            
            if (mid < end) {
                // Size of subarrays to merge
                int size1 = mid - start;
                int size2 = end - mid;
                
                // Calculate grid dimensions
                int totalThreads = size1 + size2;
                int threadsPerBlock = BLOCK_SIZE;
                int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
                
                // Launch kernel to merge these two chunks
                mergeArraysKernel<<<blocksPerGrid, threadsPerBlock>>>(
                    inputKeys + start, inputIndices + start, size1,
                    inputKeys + mid, inputIndices + mid, size2,
                    outputKeys + start, outputIndices + start,
                    isAscending
                );
            } else {
                // If there's only one chunk, copy it directly
                cudaMemcpyAsync(outputKeys + start, inputKeys + start, 
                                (end - start) * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpyAsync(outputIndices + start, inputIndices + start, 
                                (end - start) * sizeof(int), cudaMemcpyDeviceToDevice);
            }
        }
        
        // Swap input and output arrays for next iteration
        float* tempKeys = inputKeys;
        int* tempIndices = inputIndices;
        inputKeys = outputKeys;
        inputIndices = outputIndices;
        outputKeys = tempKeys;
        outputIndices = tempIndices;
        cudaDeviceSynchronize();
    }
    
    // Copy final result back to host 
    // If it's in d_temp_keys, copy from there, otherwise from d_keys
    if (inputKeys == d_temp_keys) {
        cudaMemcpy(h_keys, d_temp_keys, numRows * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_indices, d_temp_indices, numRows * sizeof(int), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(h_keys, d_keys, numRows * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_indices, d_indices, numRows * sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    // Clean up
    cudaFree(d_keys);
    cudaFree(d_indices);
    cudaFree(d_temp_keys);
    cudaFree(d_temp_indices);
}

void mergeChunksGPULongLong(long long* h_keys, int* h_indices, long long numRows, int numChunks, long long chunkSize, bool isAscending) {
    
    // We can do the full merge in one go
    // Allocate device memory for input
    long long* d_keys = nullptr;
    int* d_indices = nullptr;
    cudaMalloc(&d_keys, numRows * sizeof(long long));
    cudaMalloc(&d_indices, numRows * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_keys, h_keys, numRows * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, numRows * sizeof(int), cudaMemcpyHostToDevice);
    
    // Allocate temporary device memory for output
    long long* d_temp_keys = nullptr;
    int* d_temp_indices = nullptr;
    cudaMalloc(&d_temp_keys, numRows * sizeof(long long));
    cudaMalloc(&d_temp_indices, numRows * sizeof(int));
    
    // Create CUDA stream for merging
    cudaStream_t mergeStream;
    cudaStreamCreate(&mergeStream);
    
    // Merge chunks in parallel
    long long* inputKeys = d_keys;
    int* inputIndices = d_indices;
    long long* outputKeys = d_temp_keys;
    int* outputIndices = d_temp_indices;
    
    for (long long currentSize = chunkSize; currentSize < numRows; currentSize *= 2) {
        // For each pair of chunks at current level
        for (long long start = 0; start < numRows; start += 2 * currentSize) {
            long long mid = std::min(start + currentSize, numRows);
            long long end = std::min(start + 2 * currentSize, numRows);
            
            if (mid < end) {
                // Size of subarrays to merge
                int size1 = mid - start;
                int size2 = end - mid;
                
                // Calculate grid dimensions
                int totalThreads = size1 + size2;
                int threadsPerBlock = BLOCK_SIZE;
                int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
                
                // Launch kernel to merge these two chunks
                mergeArraysKernelLongLong<<<blocksPerGrid, threadsPerBlock, 0, mergeStream>>>(
                    inputKeys + start, inputIndices + start, size1,
                    inputKeys + mid, inputIndices + mid, size2,
                    outputKeys + start, outputIndices + start,
                    isAscending
                );
            } else {
                // If there's only one chunk, copy it directly
                cudaMemcpyAsync(outputKeys + start, inputKeys + start, 
                                (end - start) * sizeof(long long), cudaMemcpyDeviceToDevice, 
                                mergeStream);
                cudaMemcpyAsync(outputIndices + start, inputIndices + start, 
                                (end - start) * sizeof(int), cudaMemcpyDeviceToDevice, 
                                mergeStream);
            }
        }
        
        // Swap input and output arrays for next iteration
        long long* tempKeys = inputKeys;
        int* tempIndices = inputIndices;
        inputKeys = outputKeys;
        inputIndices = outputIndices;
        outputKeys = tempKeys;
        outputIndices = tempIndices;
        
        // Synchronize after each level
        cudaStreamSynchronize(mergeStream);
    }
    
    // Copy final result back to host 
    // If it's in d_temp_keys, copy from there, otherwise from d_keys
    if (inputKeys == d_temp_keys) {
        cudaMemcpy(h_keys, d_temp_keys, numRows * sizeof(long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_indices, d_temp_indices, numRows * sizeof(int), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(h_keys, d_keys, numRows * sizeof(long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_indices, d_indices, numRows * sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    // Clean up
    cudaStreamDestroy(mergeStream);
    cudaFree(d_keys);
    cudaFree(d_indices);
    cudaFree(d_temp_keys);
    cudaFree(d_temp_indices);
}
bool ExecuteSortBatch(int columnIdx, bool isAscending, Table* table, long long rowsInBatch) {
    // Get data from the table
    void** tableData = table->getData();
    int numColumns = table->getNumColumns();
    char** columnNames = table->getColumnNames();
    
    // Parse column types from column names
    std::vector<char> columnTypes(numColumns);
    for (int i = 0; i < numColumns; i++) {
        if (columnNames[i]) {
            std::string colName = columnNames[i];
            size_t typePos = colName.rfind('(');
            if (typePos != std::string::npos && colName.length() > typePos + 1) {
                columnTypes[i] = colName[typePos + 1]; 
            } else {
                columnTypes[i] = 'T'; // Default to text
            }
        } else {
            columnTypes[i] = 'T'; // Default to text
        }
    }
    
    // Create multiple CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Determine the column type for sorting
    char columnType = columnTypes[columnIdx];
    
    // Handle based on column type
    if (columnType == 'D') { // Date/DateTime column
        // Calculate chunk size for each stream
        long long chunkSize = (rowsInBatch + NUM_STREAMS - 1) / NUM_STREAMS;
        
        // Allocate host arrays for all data (pinned memory)
        long long* h_date_ints = nullptr;
        int* h_indices = nullptr;
        
        cudaError_t err = cudaMallocHost((void**)&h_date_ints, rowsInBatch * sizeof(long long));
        if (err != cudaSuccess) {
            std::cerr << "Error allocating pinned memory for date integers: " << cudaGetErrorString(err) << std::endl;
            for (int i = 0; i < NUM_STREAMS; i++) {
                cudaStreamDestroy(streams[i]);
            }
            return false;
        }
        
        err = cudaMallocHost((void**)&h_indices, rowsInBatch * sizeof(int));
        if (err != cudaSuccess) {
            std::cerr << "Error allocating pinned memory for indices: " << cudaGetErrorString(err) << std::endl;
            cudaFreeHost(h_date_ints);
            for (int i = 0; i < NUM_STREAMS; i++) {
                cudaStreamDestroy(streams[i]);
            }
            return false;
        }
        
        // Convert dates to integers on CPU
        char** dateStrings = static_cast<char**>(tableData[columnIdx]);
        #pragma omp parallel for
        for (long long i = 0; i < rowsInBatch; i++) {
            h_date_ints[i] = dateTimeToInt64(dateStrings[i]);
            h_indices[i] = i;
        }
        
        // Arrays to store device pointers for each stream
        std::vector<long long*> d_date_ints_chunks(NUM_STREAMS);
        std::vector<int*> d_indices_chunks(NUM_STREAMS);
        
        // Process data in chunks using multiple streams
        for (int s = 0; s < NUM_STREAMS; s++) {
            long long offset = s * chunkSize;
            long long currentChunkSize = std::min(chunkSize, rowsInBatch - offset);
            
            if (currentChunkSize <= 0) continue; // Skip empty chunks
            
            // Allocate device memory for this chunk
            cudaMalloc((void**)&d_date_ints_chunks[s], currentChunkSize * sizeof(long long));
            cudaMalloc((void**)&d_indices_chunks[s], currentChunkSize * sizeof(int));
            
            // Transfer chunk data to device asynchronously
            cudaMemcpyAsync(d_date_ints_chunks[s], h_date_ints + offset, 
                           currentChunkSize * sizeof(long long), 
                           cudaMemcpyHostToDevice, streams[s]);
            
            cudaMemcpyAsync(d_indices_chunks[s], h_indices + offset, 
                           currentChunkSize * sizeof(int), 
                           cudaMemcpyHostToDevice, streams[s]);
            
            // Execute merge sort on this stream for this chunk
            mergeSortGPULongLong(d_date_ints_chunks[s], d_indices_chunks[s], 
                                currentChunkSize, isAscending, streams[s]);
            
            // Copy sorted indices back to host asynchronously
            cudaMemcpyAsync(h_indices + offset, d_indices_chunks[s], 
                           currentChunkSize * sizeof(int), 
                           cudaMemcpyDeviceToHost, streams[s]);
            
            // Copy sorted keys back to host asynchronously (needed for final merge)
            cudaMemcpyAsync(h_date_ints + offset, d_date_ints_chunks[s],
                           currentChunkSize * sizeof(long long),
                           cudaMemcpyDeviceToHost, streams[s]);
        }
        
        // Synchronize all streams
        for (int s = 0; s < NUM_STREAMS; s++) {
            cudaStreamSynchronize(streams[s]);
        }
        // mergeChunksLongLong(h_date_ints, h_indices, rowsInBatch, NUM_STREAMS, chunkSize, isAscending);
        mergeChunksGPULongLong(h_date_ints, h_indices, rowsInBatch, NUM_STREAMS, chunkSize, isAscending);
        
        // Check for errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
            
            // Clean up resources
            for (int s = 0; s < NUM_STREAMS; s++) {
                if (d_date_ints_chunks[s]) cudaFree(d_date_ints_chunks[s]);
                if (d_indices_chunks[s]) cudaFree(d_indices_chunks[s]);
            }
            
            cudaFreeHost(h_date_ints);
            cudaFreeHost(h_indices);
            for (int i = 0; i < NUM_STREAMS; i++) {
                cudaStreamDestroy(streams[i]);
            }
            
            return false;
        }
        
        // Create pinned memory backups of all columns
        std::vector<void*> backupData(numColumns);
        
        for (int col = 0; col < numColumns; col++) {
            char colType = columnTypes[col];
            
            switch (colType) {
                case 'N': // Numeric
                    cudaMallocHost(&backupData[col], rowsInBatch * sizeof(float));
                    memcpy(backupData[col], tableData[col], rowsInBatch * sizeof(float));
                    break;
                    
                case 'T': // Text
                case 'D': // Date
                default:  // Default to string
                    cudaMallocHost(&backupData[col], rowsInBatch * sizeof(char*));
                    memcpy(backupData[col], tableData[col], rowsInBatch * sizeof(char*));
                    break;
            }
        }
        
        // Reorder all columns using sorted indices
        #pragma omp parallel for
        for (int col = 0; col < numColumns; col++) {
            char colType = columnTypes[col];
            
            switch (colType) {
                case 'N': // Numeric (float)
                    {
                        float* typedColumnData = static_cast<float*>(tableData[col]);
                        float* typedBackupData = static_cast<float*>(backupData[col]);
                        
                        // Reorder based on sorted indices
                        for (int i = 0; i < rowsInBatch; i++) {
                            typedColumnData[i] = typedBackupData[h_indices[i]];
                        }
                    }
                    break;
                    
                case 'T': // Text
                case 'D': // Date
                default:  // Default to string
                    {
                        char** typedColumnData = static_cast<char**>(tableData[col]);
                        char** typedBackupData = static_cast<char**>(backupData[col]);
                        
                        // Reorder based on sorted indices
                        for (int i = 0; i < rowsInBatch; i++) {
                            typedColumnData[i] = typedBackupData[h_indices[i]];
                        }
                    }
                    break;
            }
        }

        // Free the backup data
        for (int col = 0; col < numColumns; col++) {
            cudaFreeHost(backupData[col]);
        }

        // Free CUDA resources
        for (int s = 0; s < NUM_STREAMS; s++) {
            if (d_date_ints_chunks[s]) cudaFree(d_date_ints_chunks[s]);
            if (d_indices_chunks[s]) cudaFree(d_indices_chunks[s]);
        }

        cudaFreeHost(h_date_ints);
        cudaFreeHost(h_indices);
    }

        else { // Regular numeric column (float)
        // Calculate chunk size for each stream
        long long chunkSize = (rowsInBatch + NUM_STREAMS - 1) / NUM_STREAMS;
        
        // Allocate pinned memory for host arrays
        float* h_keys = nullptr;
        int* h_indices = nullptr;
        
        cudaError_t err = cudaMallocHost((void**)&h_keys, rowsInBatch * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Error allocating pinned memory for keys: " << cudaGetErrorString(err) << std::endl;
            for (int i = 0; i < NUM_STREAMS; i++) {
                cudaStreamDestroy(streams[i]);
            }
            return false;
        }
        
        err = cudaMallocHost((void**)&h_indices, rowsInBatch * sizeof(int));
        if (err != cudaSuccess) {
            std::cerr << "Error allocating pinned memory for indices: " << cudaGetErrorString(err) << std::endl;
            cudaFreeHost(h_keys);
            for (int i = 0; i < NUM_STREAMS; i++) {
                cudaStreamDestroy(streams[i]);
            }
            return false;
        }
        
        // Access the column to sort
        float* sortColumn = static_cast<float*>(tableData[columnIdx]);
        
        // Copy keys and initialize indices
        memcpy(h_keys, sortColumn, rowsInBatch * sizeof(float));
        #pragma omp parallel for
        for (int i = 0; i < rowsInBatch; i++) {
            h_indices[i] = i;
        }
        
        // Arrays to store device pointers for each stream
        std::vector<float*> d_keys_chunks(NUM_STREAMS);
        std::vector<int*> d_indices_chunks(NUM_STREAMS);
        
        // Process data in chunks using multiple streams
        for (int s = 0; s < NUM_STREAMS; s++) {
            long long offset = s * chunkSize;
            long long currentChunkSize = std::min(chunkSize, rowsInBatch - offset);
            
            if (currentChunkSize <= 0) continue; // Skip empty chunks
            
            // Allocate device memory for this chunk
            cudaMalloc((void**)&d_keys_chunks[s], currentChunkSize * sizeof(float));
            cudaMalloc((void**)&d_indices_chunks[s], currentChunkSize * sizeof(int));
            
            // Transfer chunk data to device asynchronously
            cudaMemcpyAsync(d_keys_chunks[s], h_keys + offset, 
                        currentChunkSize * sizeof(float), 
                        cudaMemcpyHostToDevice, streams[s]);
            
            cudaMemcpyAsync(d_indices_chunks[s], h_indices + offset, 
                        currentChunkSize * sizeof(int), 
                        cudaMemcpyHostToDevice, streams[s]);
            
            // Execute merge sort on this stream for this chunk
            mergeSortGPU(d_keys_chunks[s], d_indices_chunks[s], 
                        currentChunkSize, isAscending, streams[s]);
            
            // Copy sorted indices back to host asynchronously
            cudaMemcpyAsync(h_indices + offset, d_indices_chunks[s], 
                        currentChunkSize * sizeof(int), 
                        cudaMemcpyDeviceToHost, streams[s]);
            
            // Copy sorted keys back to host asynchronously (needed for final merge)
            cudaMemcpyAsync(h_keys + offset, d_keys_chunks[s],
                        currentChunkSize * sizeof(float),
                        cudaMemcpyDeviceToHost, streams[s]);
        }
        
        // Synchronize all streams
        for (int s = 0; s < NUM_STREAMS; s++) {
            cudaStreamSynchronize(streams[s]);
        }
        
        // mergeChunks(h_keys, h_indices, rowsInBatch, NUM_STREAMS, chunkSize, isAscending);
        mergeChunksGPU(h_keys, h_indices, rowsInBatch, NUM_STREAMS, chunkSize, isAscending);
        
        // Check for errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
            
            // Clean up resources
            for (int s = 0; s < NUM_STREAMS; s++) {
                if (d_keys_chunks[s]) cudaFree(d_keys_chunks[s]);
                if (d_indices_chunks[s]) cudaFree(d_indices_chunks[s]);
            }
            
            cudaFreeHost(h_keys);
            cudaFreeHost(h_indices);
            for (int i = 0; i < NUM_STREAMS; i++) {
                cudaStreamDestroy(streams[i]);
            }
            
            return false;
        }
        
        // Create pinned memory backups for all columns
        std::vector<void*> backupData(numColumns);
        
        for (int col = 0; col < numColumns; col++) {
            char colType = columnTypes[col];
            
            switch (colType) {
                case 'N': // Numeric
                    cudaMallocHost(&backupData[col], rowsInBatch * sizeof(float));
                    memcpy(backupData[col], tableData[col], rowsInBatch * sizeof(float));
                    break;
                    
                case 'T': // Text
                case 'D': // Date
                default:  // Default to string
                    cudaMallocHost(&backupData[col], rowsInBatch * sizeof(char*));
                    memcpy(backupData[col], tableData[col], rowsInBatch * sizeof(char*));
                    break;
            }
        }
        
        // Reorder all columns using sorted indices
        #pragma omp parallel for
        for (int col = 0; col < numColumns; col++) {
            char colType = columnTypes[col];
            
            switch (colType) {
                case 'N': // Numeric (float)
                    {
                        float* typedColumnData = static_cast<float*>(tableData[col]);
                        float* typedBackupData = static_cast<float*>(backupData[col]);
                        
                        // Reorder based on sorted indices
                        for (int i = 0; i < rowsInBatch; i++) {
                            typedColumnData[i] = typedBackupData[h_indices[i]];
                        }
                    }
                    break;
                    
                case 'T': // Text
                case 'D': // Date
                default:  // Default to string
                    {
                        char** typedColumnData = static_cast<char**>(tableData[col]);
                        char** typedBackupData = static_cast<char**>(backupData[col]);
                        
                        // Reorder based on sorted indices
                        for (int i = 0; i < rowsInBatch; i++) {
                            typedColumnData[i] = typedBackupData[h_indices[i]];
                        }
                    }
                    break;
            }
        }
        
        // Free the backup data
        for (int col = 0; col < numColumns; col++) {
            cudaFreeHost(backupData[col]);
        }
        
        // Free CUDA resources
        for (int s = 0; s < NUM_STREAMS; s++) {
            if (d_keys_chunks[s]) cudaFree(d_keys_chunks[s]);
            if (d_indices_chunks[s]) cudaFree(d_indices_chunks[s]);
        }
        
        cudaFreeHost(h_keys);
        cudaFreeHost(h_indices);
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return true;
}

bool ExecuteSortBatch(int columnIdx, bool isAscending, Table* table, long long rowsInBatch , int direction) {
    // Get data from the table
    void** tableData = table->getData();
    
    // Check if table data is valid
    if (!tableData) {
        std::cerr << "Error: Table data is null" << std::endl;
        return false;
    }
    
    int numColumns = table->getNumColumns();
    char** columnNames = table->getColumnNames();
    
    // Check if the specified column index is valid
    if (columnIdx < 0 || columnIdx >= numColumns) {
        std::cerr << "Error: Invalid column index: " << columnIdx << std::endl;
        return false;
    }
    
    // Check if the number of rows is valid
    if (rowsInBatch <= 0) {
        std::cerr << "Error: Invalid number of rows: " << rowsInBatch << std::endl;
        return false;
    }
    
    // Parse column types from column names
    std::vector<char> columnTypes(numColumns);
    for (int i = 0; i < numColumns; i++) {
        if (columnNames && columnNames[i]) {
            std::string colName = columnNames[i];
            size_t typePos = colName.rfind('(');
            if (typePos != std::string::npos && colName.length() > typePos + 1) {
                columnTypes[i] = colName[typePos + 1]; 
            } else {
                columnTypes[i] = 'N'; 
            }
        } else {
            columnTypes[i] = 'N';
        }
    }
    
    // Determine the column type for sorting
    char columnType = columnTypes[columnIdx];

    // Check if the column data exists
    if (!tableData[columnIdx]) {
        std::cerr << "Error: Data for column " << columnIdx << " is null" << std::endl;
        return false;
    }
    
    // Create host indices vector with error checking
    std::vector<int> indices(rowsInBatch);
    for (int i = 0; i < rowsInBatch; i++) {
        indices[i] = i;
    }
    
    // Use std::vector first for safety, then convert to thrust vectors
    thrust::host_vector<int> h_indices(indices.begin(), indices.end());
    
    try {
        // Sort based on column type
        if (columnType == 'N') { // Numeric (float)
            // Access the column to sort with explicit casting
            float* sortColumn = static_cast<float*>(tableData[columnIdx]);
            
            // Safely copy the data to std::vector first
            std::vector<float> keys(rowsInBatch);
            for (int i = 0; i < rowsInBatch; i++) {
                keys[i] = sortColumn[i];
            }
            
            // Then create thrust vectors
            thrust::host_vector<float> h_keys(keys.begin(), keys.end());
            thrust::device_vector<float> d_keys = h_keys;
            thrust::device_vector<int> d_indices = h_indices;
            
            // Sort by keys, either ascending or descending
            if (isAscending) {
                thrust::sort_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_indices.begin());
            } else {
                thrust::sort_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_indices.begin(), 
                                   thrust::greater<float>());
            }
            
            // Copy sorted indices back to host
            thrust::copy(d_indices.begin(), d_indices.end(), h_indices.begin());
        }
        else if (columnType == 'D') { // Date
            // For date columns, convert dates to integers for sorting
            char** dateStrings = static_cast<char**>(tableData[columnIdx]);
            
            // Create vector for converted date keys
            std::vector<long long> dateKeys(rowsInBatch);
            
            // Convert dates to integer representation
            for (int i = 0; i < rowsInBatch; i++) {
                dateKeys[i] = dateStrings[i] ? dateTimeToInt64(dateStrings[i]) : 0;
            }
            
            // Create thrust vectors
            thrust::host_vector<long long> h_date_keys(dateKeys.begin(), dateKeys.end());
            thrust::device_vector<long long> d_date_keys = h_date_keys;
            thrust::device_vector<int> d_indices = h_indices;
            
            // Sort by keys, either ascending or descending
            if (isAscending) {
                thrust::sort_by_key(thrust::device, d_date_keys.begin(), d_date_keys.end(), d_indices.begin());
            } else {
                thrust::sort_by_key(thrust::device, d_date_keys.begin(), d_date_keys.end(), d_indices.begin(),
                                   thrust::greater<long long>());
            }
            
            // Copy sorted indices back to host
            thrust::copy(d_indices.begin(), d_indices.end(), h_indices.begin());
        }
        else {
            std::cerr << "Unknown column type" << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during sorting: " << e.what() << std::endl;
        return false;
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    std::vector<void*> backupData(numColumns, nullptr);
    
    try {
        for (int col = 0; col < numColumns; col++) {
            if (!tableData[col]) {
                std::cerr << "Warning: Data for column " << col << " is null, skipping backup" << std::endl;
                continue;
            }
            
            char colType = columnTypes[col];
            
            switch (colType) {
                case 'N': // Numeric (float)
                    cudaMallocHost(&backupData[col], rowsInBatch * sizeof(float));
                    if (!backupData[col]) {
                        throw std::runtime_error("Failed to allocate backup memory for numeric column");
                    }
                    memcpy(backupData[col], tableData[col], rowsInBatch * sizeof(float));
                    break;
                    
                case 'T': // Text
                case 'D': // Date
                    // For text and date columns, need to copy pointers to strings
                    cudaMallocHost(&backupData[col], rowsInBatch * sizeof(char*));
                    if (!backupData[col]) {
                        throw std::runtime_error("Failed to allocate backup memory for text/date column");
                    }
                    memcpy(backupData[col], tableData[col], rowsInBatch * sizeof(char*));
                    break;
                    
                default:
                    // Default to float for unknown types
                    cudaMallocHost(&backupData[col], rowsInBatch * sizeof(float));
                    if (!backupData[col]) {
                        throw std::runtime_error("Failed to allocate backup memory for default column");
                    }
                    memcpy(backupData[col], tableData[col], rowsInBatch * sizeof(float));
                    break;
            }
        }
        
        // Now use the sorted indices to reorder all columns
        for (int col = 0; col < numColumns; col++) {
            if (!tableData[col] || !backupData[col]) {
                std::cerr << "Warning: Data for column " << col << " is null, skipping reordering" << std::endl;
                continue;
            }
            
            char colType = columnTypes[col];
            
            switch (colType) {
                case 'N': // Numeric (float)
                    {
                        float* typedColumnData = static_cast<float*>(tableData[col]);
                        float* typedBackupData = static_cast<float*>(backupData[col]);
                        
                        // Reorder based on sorted indices
                        for (int i = 0; i < rowsInBatch; i++) {
                            int srcIdx = h_indices[i];
                            if (srcIdx >= 0 && srcIdx < rowsInBatch) {
                                typedColumnData[i] = typedBackupData[srcIdx];
                            }
                        }
                    }
                    break;
                    
                case 'T': // Text
                case 'D': // Date
                    {
                        char** typedColumnData = static_cast<char**>(tableData[col]);
                        char** typedBackupData = static_cast<char**>(backupData[col]);
                        
                        // Reorder based on sorted indices
                        for (int i = 0; i < rowsInBatch; i++) {
                            int srcIdx = h_indices[i];
                            if (srcIdx >= 0 && srcIdx < rowsInBatch) {
                                typedColumnData[i] = typedBackupData[srcIdx];
                            }
                        }
                    }
                    break;
                    
                default:
                    // Default to float
                    {
                        float* typedColumnData = static_cast<float*>(tableData[col]);
                        float* typedBackupData = static_cast<float*>(backupData[col]);
                        
                        // Reorder based on sorted indices
                        for (int i = 0; i < rowsInBatch; i++) {
                            int srcIdx = h_indices[i];
                            if (srcIdx >= 0 && srcIdx < rowsInBatch) {
                                typedColumnData[i] = typedBackupData[srcIdx];
                            }
                        }
                    }
                    break;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during reordering: " << e.what() << std::endl;
        
        // Free the backup data
        for (int col = 0; col < numColumns; col++) {
            if (backupData[col]) {
                cudaFreeHost(backupData[col]);
            }
        }
        return false;
    }
    
    // Free the backup data
    for (int col = 0; col < numColumns; col++) {
        if (backupData[col]) {
            cudaFreeHost(backupData[col]);
        }
    }
    
    return true;
}

float ExecuteMinMaxFloat(int columnIdx, bool findMin, Table* last_table_scanned_h, long long numRows) {
    // Get the data for the specified column
    void* columnData = last_table_scanned_h->getData()[columnIdx];
    float* h_data = static_cast<float*>(columnData);
    
    cudaStream_t streams[NUM_STREAMS];
    
    // Create streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Thread and block configuration
    int threadsPerBlock = BLOCK_SIZE;
    
    // Calculate chunk size for each stream
    long long streamSize = (numRows + NUM_STREAMS - 1) / NUM_STREAMS;
    
    // Calculate blocks needed for each stream chunk
    int* blocksPerStream = new int[NUM_STREAMS];
    int totalBlocks = 0;
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        long long offset = i * streamSize;
        long long currentSize = std::min(streamSize, numRows - offset);
        if (currentSize <= 0) break;
        
        blocksPerStream[i] = (currentSize + threadsPerBlock - 1) / threadsPerBlock;
        totalBlocks += blocksPerStream[i];
    }
    
    // Allocate pinned memory for results
    float *h_block_results = nullptr;
    cudaMallocHost(&h_block_results, totalBlocks * sizeof(float));
    
    // Allocate GPU memory
    float *d_data = nullptr;
    float *d_block_results = nullptr;
    cudaMalloc(&d_data, numRows * sizeof(float));
    cudaMalloc(&d_block_results, totalBlocks * sizeof(float));
    
    // Copy data to GPU memory (can be done in one operation or split)
    cudaMemcpy(d_data, h_data, numRows * sizeof(float), cudaMemcpyHostToDevice);
    
    // Process data in streams
    int blockOffset = 0;
    for (int i = 0; i < NUM_STREAMS; i++) {
        long long dataOffset = i * streamSize;
        long long currentSize = std::min(streamSize, numRows - dataOffset);
        if (currentSize <= 0) break;
        
        int currentBlocks = blocksPerStream[i];
        
        // Launch kernel on stream
        if (findMin) {
            findWarpMin<<<currentBlocks, threadsPerBlock, 0, streams[i]>>>(
                d_data + dataOffset, 
                d_block_results + blockOffset, 
                currentSize);
        } else {
            findWarpMax<<<currentBlocks, threadsPerBlock, 0, streams[i]>>>(
                d_data + dataOffset, 
                d_block_results + blockOffset, 
                currentSize);
        }
        
        blockOffset += currentBlocks;
    }
    
    // Copy all results back to host with a single operation
    cudaMemcpy(h_block_results, d_block_results, totalBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Synchronize to ensure all operations are complete
    cudaDeviceSynchronize();
    
    // Perform final reduction on CPU
    float result = findMin ? FLT_MAX : -FLT_MAX;
    for (int i = 0; i < totalBlocks; i++) {
        if (findMin) {
            result = fminf(result, h_block_results[i]);
        } else {
            result = fmaxf(result, h_block_results[i]);
        }
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "\nCUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Clean up resources
    cudaFreeHost(h_block_results);
    cudaFree(d_data);
    cudaFree(d_block_results);
    delete[] blocksPerStream;
    
    // Destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    return result;
}

long long ExecuteMinMaxDate(int columnIdx, bool findMin, Table* last_table_scanned_h, long long numRows) {
    // Get the data for the specified column
    void* columnData = last_table_scanned_h->getData()[columnIdx];
    char** dateStrings = static_cast<char**>(columnData);
    
    if (numRows == 0) {
        std::cout << "No data to process for " << (findMin ? "MIN" : "MAX") << " DATE operation" << std::endl;
        return findMin ? LLONG_MAX : LLONG_MIN;
    }
    
    cudaStream_t streams[NUM_STREAMS];
    
    // Create streams first
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaError_t streamErr = cudaStreamCreate(&streams[i]);
        if (streamErr != cudaSuccess) {
            std::cerr << "Stream creation error: " << cudaGetErrorString(streamErr) << std::endl;
            return findMin ? LLONG_MAX : LLONG_MIN;
        }
    }
    
    // Thread and block configuration
    int threadsPerBlock = BLOCK_SIZE;
    
    // Calculate chunk size for each stream
    long long streamSize = (numRows + NUM_STREAMS - 1) / NUM_STREAMS;
    
    // Calculate blocks needed for each stream chunk
    int* blocksPerStream = new int[NUM_STREAMS];
    int totalBlocks = 0;
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        long long offset = i * streamSize;
        if (offset >= numRows) break;
        
        long long currentSize = std::min(streamSize, numRows - offset);
        blocksPerStream[i] = (currentSize + threadsPerBlock - 1) / threadsPerBlock;
        totalBlocks += blocksPerStream[i];
    }
    
    // Create a contiguous array of date strings in pinned memory
    char* h_contiguous_strings = nullptr;
    cudaError_t memErr = cudaMallocHost(&h_contiguous_strings, numRows * MAX_DATETIME * sizeof(char));
    if (memErr != cudaSuccess || h_contiguous_strings == nullptr) {
        std::cerr << "Pinned memory allocation error: " << cudaGetErrorString(memErr) << std::endl;
        delete[] blocksPerStream;
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
        return findMin ? LLONG_MAX : LLONG_MIN;
    }
    
    // Copy date strings to contiguous pinned memory with bounds checking
    for (size_t i = 0; i < numRows; i++) {
        if (dateStrings[i] != nullptr) {
            strncpy(h_contiguous_strings + i * MAX_DATETIME, dateStrings[i], MAX_DATETIME - 1);
        } else {
            // Handle NULL entries safely
            memset(h_contiguous_strings + i * MAX_DATETIME, 0, MAX_DATETIME);
        }
        h_contiguous_strings[i * MAX_DATETIME + MAX_DATETIME - 1] = '\0'; // Ensure null termination
    }
    
    // Allocate pinned memory for block results
    long long* h_block_results = nullptr;
    memErr = cudaMallocHost(&h_block_results, totalBlocks * sizeof(long long));
    if (memErr != cudaSuccess || h_block_results == nullptr) {
        std::cerr << "Block results allocation error: " << cudaGetErrorString(memErr) << std::endl;
        cudaFreeHost(h_contiguous_strings);
        delete[] blocksPerStream;
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
        return findMin ? LLONG_MAX : LLONG_MIN;
    }
    
    // Setup initial values for min/max results
    long long initialValue = findMin ? LLONG_MAX : LLONG_MIN;
    for (int i = 0; i < totalBlocks; i++) {
        h_block_results[i] = initialValue;
    }
    
    // Allocate device memory with error checking
    char* d_date_strings = nullptr;
    long long* d_dates = nullptr;
    long long* d_block_results = nullptr;
    
    if (cudaMalloc(&d_date_strings, numRows * MAX_DATETIME * sizeof(char)) != cudaSuccess ||
        cudaMalloc(&d_dates, numRows * sizeof(long long)) != cudaSuccess ||
        cudaMalloc(&d_block_results, totalBlocks * sizeof(long long)) != cudaSuccess) {
        
        std::cerr << "Device memory allocation error" << std::endl;
        if (d_date_strings) cudaFree(d_date_strings);
        if (d_dates) cudaFree(d_dates);
        if (d_block_results) cudaFree(d_block_results);
        cudaFreeHost(h_contiguous_strings);
        cudaFreeHost(h_block_results);
        delete[] blocksPerStream;
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
        return findMin ? LLONG_MAX : LLONG_MIN;
    }
    
    // Initialize d_block_results
    cudaMemcpy(d_block_results, h_block_results, 
               totalBlocks * sizeof(long long), 
               cudaMemcpyHostToDevice);
    
    // Process data in streams with correct block offset tracking
    int blockOffset = 0;
    for (int i = 0; i < NUM_STREAMS; i++) {
        long long dataOffset = i * streamSize;
        if (dataOffset >= numRows) break;
        
        long long currentSize = std::min(streamSize, numRows - dataOffset);
        int currentBlocks = blocksPerStream[i];
        
        // Asynchronously copy chunk of string data to GPU
        cudaMemcpyAsync(d_date_strings + dataOffset * MAX_DATETIME, 
                        h_contiguous_strings + dataOffset * MAX_DATETIME, 
                        currentSize * MAX_DATETIME * sizeof(char), 
                        cudaMemcpyHostToDevice, 
                        streams[i]);
        
        // Check for errors after memory copy
        cudaError_t copyErr = cudaGetLastError();
        if (copyErr != cudaSuccess) {
            std::cerr << "Memory copy error in stream " << i << ": " << cudaGetErrorString(copyErr) << std::endl;
            continue;
        }
        
        // Convert date strings to int64 representation in this stream
        convertDateTimesToInt64Kernel<<<currentBlocks, threadsPerBlock, 0, streams[i]>>>(
            d_date_strings + dataOffset * MAX_DATETIME, 
            d_dates + dataOffset, 
            currentSize, 
            MAX_DATETIME, 
            findMin);
            
        // Check for kernel launch errors
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            std::cerr << "Conversion kernel error in stream " << i << ": " << cudaGetErrorString(kernelErr) << std::endl;
            continue;
        }
        
        // Find min/max date for each block in this stream with correct result offset
        if (findMin) {
            findWarpMinDate<<<currentBlocks, threadsPerBlock, 0, streams[i]>>>(
                d_dates + dataOffset, 
                d_block_results + blockOffset,  // Use blockOffset instead of i * blocksPerStream
                currentSize);
        } else {
            findWarpMaxDate<<<currentBlocks, threadsPerBlock, 0, streams[i]>>>(
                d_dates + dataOffset, 
                d_block_results + blockOffset,  // Use blockOffset instead of i * blocksPerStream
                currentSize);
        }
        
        // Check for kernel launch errors
        kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            std::cerr << "Min/Max kernel error in stream " << i << ": " << cudaGetErrorString(kernelErr) << std::endl;
            continue;
        }
        
        // Update block offset for next stream
        blockOffset += currentBlocks;
    }
    
    // Copy all results back to host with a single operation
    cudaMemcpy(h_block_results, d_block_results, totalBlocks * sizeof(long long), cudaMemcpyDeviceToHost);
    
    // Synchronize to ensure all operations are complete
    cudaDeviceSynchronize();
    
    // CPU performs final reduction on block results
    long long gpu_result = initialValue;
    for (int i = 0; i < totalBlocks; i++) {
        long long block_val = h_block_results[i];
        
        if (block_val != initialValue && block_val > 0) {
            if ((findMin && (gpu_result == LLONG_MAX || block_val < gpu_result)) ||
                (!findMin && block_val > gpu_result)) {
                gpu_result = block_val;
            }
        }
    }
    
    // Clean up resources
    cudaFree(d_date_strings);
    cudaFree(d_dates);
    cudaFree(d_block_results);
    cudaFreeHost(h_contiguous_strings);
    cudaFreeHost(h_block_results);
    delete[] blocksPerStream;
    
    // Destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    return gpu_result;
}

double ExecuteSumFloat(int columnIdx, Table* last_table_scanned_h, long long numRows) {
    // Get the data for the specified column
    void* columnData = last_table_scanned_h->getData()[columnIdx];
    float* original_data = static_cast<float*>(columnData);
    
    if (numRows == 0) {
        std::cout << "No data to process for SUM operation" << std::endl;
        return 0;
    }
    
    cudaStream_t streams[NUM_STREAMS];
    
    // Create streams with error checking
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaError_t streamErr = cudaStreamCreate(&streams[i]);
        if (streamErr != cudaSuccess) {
            std::cerr << "Stream creation error: " << cudaGetErrorString(streamErr) << std::endl;
            // Clean up previously created streams
            for (int j = 0; j < i; j++) {
                cudaStreamDestroy(streams[j]);
            }
            return 0.0;
        }
    }
    
    // Thread and block configuration
    int threadsPerBlock = BLOCK_SIZE;
    int totalBlocks = (numRows + threadsPerBlock - 1) / threadsPerBlock;
    
    // Calculate chunk size for each stream
    long long streamSize = (numRows + NUM_STREAMS - 1) / NUM_STREAMS;
    int blocksPerStream = (streamSize + threadsPerBlock - 1) / threadsPerBlock;
    
    // Ensure totalBlocks is at least the sum of all blocksPerStream
    totalBlocks = std::max(totalBlocks, blocksPerStream * NUM_STREAMS);
    
    // Allocate pinned memory for input data
    float* h_data = nullptr;
    cudaError_t memErr = cudaMallocHost(&h_data, numRows * sizeof(float));
    if (memErr != cudaSuccess || h_data == nullptr) {
        std::cerr << "Pinned memory allocation error for input data: " << cudaGetErrorString(memErr) << std::endl;
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
        return 0.0;
    }
    
    // Copy to pinned memory
    memcpy(h_data, original_data, numRows * sizeof(float));
    
    // Allocate pinned memory for block sums
    double* h_blockSums = nullptr;
    memErr = cudaMallocHost(&h_blockSums, totalBlocks * sizeof(double));
    if (memErr != cudaSuccess || h_blockSums == nullptr) {
        std::cerr << "Pinned memory allocation error for block sums: " << cudaGetErrorString(memErr) << std::endl;
        cudaFreeHost(h_data);
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
        return 0.0;
    }
    
    // Initialize block sums to zero
    for (int i = 0; i < totalBlocks; i++) {
        h_blockSums[i] = 0.0;
    }
    
    // Allocate device memory with error checking
    float* d_data = nullptr;
    double* d_blockSums = nullptr;
    
    if (cudaMalloc(&d_data, numRows * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_blockSums, totalBlocks * sizeof(double)) != cudaSuccess) {
        
        std::cerr << "Device memory allocation error" << std::endl;
        if (d_data) cudaFree(d_data);
        if (d_blockSums) cudaFree(d_blockSums);
        cudaFreeHost(h_data);
        cudaFreeHost(h_blockSums);
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
        return 0.0;
    }
    
    // Initialize d_blockSums with async transfer
    cudaMemcpyAsync(d_blockSums, h_blockSums, 
                   totalBlocks * sizeof(double), 
                   cudaMemcpyHostToDevice, 
                   streams[0]);
    
    // Process data in streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        long long offset = i * streamSize;
        
        // Ensure we don't process beyond the end of the data
        if (offset >= numRows) continue;
        
        long long currentSize = std::min(streamSize, numRows - offset);
        int currentBlocks = (currentSize + threadsPerBlock - 1) / threadsPerBlock;
        
        // Verify that we're within bounds for all arrays
        if (offset + currentSize > numRows || i * blocksPerStream + currentBlocks > totalBlocks) {
            std::cerr << "Stream " << i << " would access out of bounds memory" << std::endl;
            continue;
        }
        
        // Asynchronously copy chunk of data to GPU
        cudaMemcpyAsync(d_data + offset, 
                       h_data + offset, 
                       currentSize * sizeof(float), 
                       cudaMemcpyHostToDevice, 
                       streams[i]);
        
        // Check for errors after memory copy
        cudaError_t copyErr = cudaGetLastError();
        if (copyErr != cudaSuccess) {
            std::cerr << "Memory copy error in stream " << i << ": " << cudaGetErrorString(copyErr) << std::endl;
            continue;
        }
        
        // Launch kernel to calculate sum for this chunk
        sumKernel<<<currentBlocks, threadsPerBlock, 0, streams[i]>>>(
            d_data + offset, 
            d_blockSums + i * blocksPerStream, 
            currentSize);
        
        // Check for kernel launch errors
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            std::cerr << "Sum kernel error in stream " << i << ": " << cudaGetErrorString(kernelErr) << std::endl;
            continue;
        }
        
        // Asynchronously copy results back to host
        cudaMemcpyAsync(h_blockSums + i * blocksPerStream, 
                       d_blockSums + i * blocksPerStream, 
                       currentBlocks * sizeof(double), 
                       cudaMemcpyDeviceToHost, 
                       streams[i]);
    }
    
    // Synchronize all streams before final reduction
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // CPU performs final reduction
    double gpu_sum = 0.0;
    for (int i = 0; i < totalBlocks; i++) {
        gpu_sum += h_blockSums[i];
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "\nCUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Clean up resources
    cudaFree(d_data);
    cudaFree(d_blockSums);
    cudaFreeHost(h_data);
    cudaFreeHost(h_blockSums);
    
    // Destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    return gpu_sum;
}
unsigned int ExecuteCountString(int columnIdx, Table* last_table_scanned_h,long long numRows) {
    // Get the data for the specified column
    void* columnData = last_table_scanned_h->getData()[columnIdx];
    char** stringData = static_cast<char**>(columnData);
    
    if (numRows == 0) {
        std::cout << "No data to process for COUNT operation" << std::endl;
        return 0;
    }
    
    // Determine maximum string length
    int maxStringLen = MAX_VAR_CHAR;  // Use a predefined constant or calculate dynamically
    
    // Create a contiguous array of strings for GPU processing
    char* h_contiguous_strings = (char*)malloc(numRows * maxStringLen * sizeof(char));
    // Initialize memory to ensure null-termination
    memset(h_contiguous_strings, 0, numRows * maxStringLen * sizeof(char));
    
    // Copy strings to contiguous array
    for (size_t i = 0; i < numRows; i++) {
        if (stringData[i] != nullptr) {
            strncpy(h_contiguous_strings + i * maxStringLen, stringData[i], maxStringLen - 1);
        }
        // Ensure null termination
        h_contiguous_strings[i * maxStringLen + maxStringLen - 1] = '\0';
    }
    
    // ================= GPU Implementation =================
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    float gpu_time = 0.0f;
    
    // Allocate device memory
    char* d_strings = nullptr;
    unsigned int* d_count = nullptr;
    
    // Calculate kernel launch parameters
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (numRows + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_strings, numRows * maxStringLen * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(unsigned int));
    
    // Copy data from host to device (not included in timing)
    cudaMemcpy(d_strings, h_contiguous_strings, numRows * maxStringLen * sizeof(char), cudaMemcpyHostToDevice);
    
    // Initialize count to 0
    unsigned int zero = 0;
    cudaMemcpy(d_count, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    // Start GPU timing
    cudaEventRecord(gpu_start, 0);
    
    // Launch kernel
    countNonNullStringsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_strings, numRows, maxStringLen, d_count);
    
    // Stop GPU timing
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
    
    // Copy the result back to the host
    unsigned int gpu_count;
    cudaMemcpy(&gpu_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "\nCUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Clean up
    free(h_contiguous_strings);
    cudaFree(d_strings);
    cudaFree(d_count);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    
    return gpu_count;
}

unsigned int ExecuteCountFloat(int columnIdx, Table* last_table_scanned_h, long long numRows) {
    // Get the data for the specified column
    void* columnData = last_table_scanned_h->getData()[columnIdx];
    float* original_data = static_cast<float*>(columnData);
    
    if (numRows == 0) {
        std::cout << "No data to process for COUNT operation" << std::endl;
        return 0;
    }
    
    cudaStream_t streams[NUM_STREAMS];
    
    // Create streams with error checking
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaError_t streamErr = cudaStreamCreate(&streams[i]);
        if (streamErr != cudaSuccess) {
            std::cerr << "Stream creation error: " << cudaGetErrorString(streamErr) << std::endl;
            // Clean up previously created streams
            for (int j = 0; j < i; j++) {
                cudaStreamDestroy(streams[j]);
            }
            return 0;
        }
    }
    
    // Thread and block configuration
    int threadsPerBlock = BLOCK_SIZE;
    
    // Calculate chunk size for each stream
    long long streamSize = (numRows + NUM_STREAMS - 1) / NUM_STREAMS;
    
    // Allocate pinned memory for input data
    float* h_data = nullptr;
    cudaError_t memErr = cudaMallocHost(&h_data, numRows * sizeof(float));
    if (memErr != cudaSuccess || h_data == nullptr) {
        std::cerr << "Pinned memory allocation error for input data: " << cudaGetErrorString(memErr) << std::endl;
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
        return 0;
    }
    
    // Copy to pinned memory
    memcpy(h_data, original_data, numRows * sizeof(float));
    
    // Allocate device memory for data and per-stream counts
    float* d_floats = nullptr;
    unsigned int* d_counts = nullptr;
    
    if (cudaMalloc(&d_floats, numRows * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_counts, NUM_STREAMS * sizeof(unsigned int)) != cudaSuccess) {
        
        std::cerr << "Device memory allocation error" << std::endl;
        if (d_floats) cudaFree(d_floats);
        if (d_counts) cudaFree(d_counts);
        cudaFreeHost(h_data);
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
        return 0;
    }
    
    // Initialize all stream counts to 0
    unsigned int zeros[NUM_STREAMS] = {0};
    cudaMemcpy(d_counts, zeros, NUM_STREAMS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    // Allocate pinned memory for results
    unsigned int* h_counts = nullptr;
    cudaMallocHost(&h_counts, NUM_STREAMS * sizeof(unsigned int));
    
    // Setup timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, streams[0]);
    
    // Process data in streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        long long offset = i * streamSize;
        
        // Skip if this stream has no data to process
        if (offset >= numRows) continue;
        
        long long currentSize = std::min(streamSize, numRows - offset);
        int currentBlocks = (currentSize + threadsPerBlock - 1) / threadsPerBlock;
        
        // Asynchronously copy chunk of data to GPU
        cudaMemcpyAsync(d_floats + offset, 
                       h_data + offset, 
                       currentSize * sizeof(float), 
                       cudaMemcpyHostToDevice, 
                       streams[i]);
        
        // Launch kernel for this stream with its own count
        countNonNullFloatsKernel<<<currentBlocks, threadsPerBlock, 0, streams[i]>>>(
            d_floats + offset, 
            currentSize, 
            &d_counts[i]);
        
        // Asynchronously copy stream result back to host
        cudaMemcpyAsync(&h_counts[i], 
                       &d_counts[i], 
                       sizeof(unsigned int), 
                       cudaMemcpyDeviceToHost, 
                       streams[i]);
    }
    
    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // Record end time
    cudaEventRecord(stop, streams[0]);
    cudaEventSynchronize(stop);
    
    // Calculate timing
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Sum up the counts from all streams
    unsigned int totalCount = 0;
    for (int i = 0; i < NUM_STREAMS; i++) {
        totalCount += h_counts[i];
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "\nCUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Clean up
    cudaFree(d_floats);
    cudaFree(d_counts);
    cudaFreeHost(h_data);
    cudaFreeHost(h_counts);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    return totalCount;
}

void ExecuteAggregateFunction(const std::string& function, int columnIdx, Table* last_table_scanned_h, const std::string& AGG_OUT_PATH, bool IS_GPU) {
    if (columnIdx >= last_table_scanned_h->getNumColumns()) {
        std::cerr << "Error: Column index " << columnIdx << " out of bounds" << std::endl;
        return;
    }
    
    std::string columnName = last_table_scanned_h->getColumnNames()[columnIdx];
    std::string columnType;
    
    size_t typePos = columnName.find('(');
    if (typePos != std::string::npos && columnName.length() > typePos + 2) {
        char typeChar = columnName[typePos + 1];
        if (typeChar == 'N') {
            columnType = "FLOAT";
        } else if (typeChar == 'D') {
            columnType = "DATE";
        } else if (typeChar == 'T') {
            columnType = "TEXT";
        }
    }
    
    if (function == "max") {
        if (columnType == "FLOAT") {
            
            int numBatches = last_table_scanned_h->getNumBatches();
            long long totalRows = last_table_scanned_h->getNumRows();
            std::vector<float> batchResults;
            
            for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
                long long rowsInBatch;
                if (batchIdx == numBatches - 1) {
                    rowsInBatch = totalRows % BATCH_SIZE;
                    if (rowsInBatch == 0 && totalRows > 0) {
                        rowsInBatch = BATCH_SIZE;
                    }
                } else {
                    rowsInBatch = BATCH_SIZE;
                }
                
                if (batchIdx > 0) last_table_scanned_h->getTableBatch(batchIdx);
                
                float batchMax = 0;
                if (IS_GPU) {
                    batchMax = ExecuteMinMaxFloat(columnIdx, false, last_table_scanned_h, rowsInBatch);
                }
                else {
                    batchMax = ExecuteMinMaxFloatCPU(columnIdx, false, last_table_scanned_h, rowsInBatch);
                }
                batchResults.push_back(batchMax);
            }
            
            float maxValue = -FLT_MAX;
            for (float value : batchResults) {
                maxValue = fmaxf(maxValue, value);
            }
            WriteAggregationToCSV("max", columnName, &maxValue, "FLOAT", "FLOAT",AGG_OUT_PATH);
        } 
        else if (columnType == "DATE") {
            int numBatches = last_table_scanned_h->getNumBatches();
            long long totalRows = last_table_scanned_h->getNumRows();
            std::vector<long long> batchResults;
            
            for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
                long long rowsInBatch;
                if (batchIdx == numBatches - 1) {
                    rowsInBatch = totalRows % BATCH_SIZE;
                    if (rowsInBatch == 0 && totalRows > 0) {
                        rowsInBatch = BATCH_SIZE;
                    }
                } else {
                    rowsInBatch = BATCH_SIZE;
                }
                
                if (batchIdx > 0) last_table_scanned_h->getTableBatch(batchIdx);
                long long batchMax = 0;
                if (IS_GPU) {
                    batchMax = ExecuteMinMaxDate(columnIdx, false, last_table_scanned_h, rowsInBatch);
                }
                else {
                    batchMax = ExecuteMinMaxDateCPU(columnIdx, false, last_table_scanned_h, rowsInBatch);
                }
                batchResults.push_back(batchMax);
            }
            
            long long maxDate = LLONG_MIN; 
            for (long long value : batchResults) {
                maxDate = std::max(maxDate, value);
            }
            
            char dateStr[MAX_DATETIME];
            int64ToDateTime(maxDate, dateStr);
            WriteAggregationToCSV("max", columnName, &dateStr, "DATE", "FLOAT",AGG_OUT_PATH);
        } 
        else {
            std::cerr << "Error: MAX operation not supported for column type " << columnType << std::endl;
        }
    } else if (function == "min") {
        if (columnType == "FLOAT") {
            int numBatches = last_table_scanned_h->getNumBatches();
            long long totalRows = last_table_scanned_h->getNumRows();
            std::vector<float> batchResults;
            
            for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
                long long rowsInBatch;
                if (batchIdx == numBatches - 1) {
                    rowsInBatch = totalRows % BATCH_SIZE;
                    if (rowsInBatch == 0 && totalRows > 0) {
                        rowsInBatch = BATCH_SIZE;
                    }
                } else {
                    rowsInBatch = BATCH_SIZE;
                }
                
                if (batchIdx > 0) last_table_scanned_h->getTableBatch(batchIdx);
                
                float batchMin = 0;
                if (IS_GPU) {
                    batchMin = ExecuteMinMaxFloat(columnIdx, true, last_table_scanned_h, rowsInBatch);
                }
                else {
                    batchMin = ExecuteMinMaxFloatCPU(columnIdx, true, last_table_scanned_h, rowsInBatch);
                }
                batchResults.push_back(batchMin);
            }
            float minValue = FLT_MAX;
            for (float value : batchResults) {
                minValue = fminf(minValue, value);
            }
            WriteAggregationToCSV("min", columnName, &minValue, "FLOAT", "FLOAT",AGG_OUT_PATH);
            
        } else if (columnType == "DATE") {
            int numBatches = last_table_scanned_h->getNumBatches();
            long long totalRows = last_table_scanned_h->getNumRows();
            std::vector<long long> batchResults;
            
            for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
                long long rowsInBatch;
                if (batchIdx == numBatches - 1) {
                    rowsInBatch = totalRows % BATCH_SIZE;
                    if (rowsInBatch == 0 && totalRows > 0) {
                        rowsInBatch = BATCH_SIZE;
                    }
                } else {
                    rowsInBatch = BATCH_SIZE;
                }
                
                if (batchIdx > 0) last_table_scanned_h->getTableBatch(batchIdx);
               
                long long batchMin = 0;
                if (IS_GPU) {
                    batchMin = ExecuteMinMaxDate(columnIdx, true, last_table_scanned_h, rowsInBatch);
                }
                else {
                    batchMin = ExecuteMinMaxDateCPU(columnIdx, true, last_table_scanned_h, rowsInBatch);
                }
                batchResults.push_back(batchMin);
            }
            
            long long minDate = LLONG_MAX; 
            for (long long value : batchResults) {
                minDate = std::min(minDate, value);
            }
            
            char dateStr[MAX_DATETIME];
            int64ToDateTime(minDate, dateStr);
            WriteAggregationToCSV("min", columnName, &dateStr, "DATE", "FLOAT",AGG_OUT_PATH);
        } 
        else {
            std::cerr << "Error: MIN operation not supported for column type " << columnType << std::endl;
        }
    } else if (function == "sum") {
        if (columnType == "FLOAT") {

            int numBatches = last_table_scanned_h->getNumBatches();
            long long totalRows = last_table_scanned_h->getNumRows();
            std::vector<double> batchResults;
            
            for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
                long long rowsInBatch;
                if (batchIdx == numBatches - 1) {
                    rowsInBatch = totalRows % BATCH_SIZE;
                    if (rowsInBatch == 0 && totalRows > 0) {
                        rowsInBatch = BATCH_SIZE;
                    }
                } else {
                    rowsInBatch = BATCH_SIZE;
                }
                
                if (batchIdx > 0) last_table_scanned_h->getTableBatch(batchIdx);
                
                double batchSum = 0;
                if (IS_GPU) {
                    batchSum = ExecuteSumFloat(columnIdx, last_table_scanned_h, rowsInBatch);
                }
                else {
                    batchSum = ExecuteSumFloatCPU(columnIdx, last_table_scanned_h, rowsInBatch);
                }
                batchResults.push_back(batchSum);
            }
            
            double finalSum = 0.0;
            for (int i = 0; i < batchResults.size(); i++) {
                double batch_val = batchResults[i];
                finalSum += batch_val;
            }
            WriteAggregationToCSV("sum", columnName, &finalSum, "FLOAT", "DOUBLE", AGG_OUT_PATH);
        } 
        else {
            std::cerr << "Error: SUM operation not supported for column type " << columnType << std::endl;
        }
    } else if (function == "count") {
        if (columnType == "TEXT" || columnType == "DATE") {
            int numBatches = last_table_scanned_h->getNumBatches();
            long long totalRows = last_table_scanned_h->getNumRows();
            std::vector<unsigned int> batchResults;
            
            for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
                long long rowsInBatch;
                if (batchIdx == numBatches - 1) {
                    rowsInBatch = totalRows % BATCH_SIZE;
                    if (rowsInBatch == 0 && totalRows > 0) {
                        rowsInBatch = BATCH_SIZE;
                    }
                } else {
                    rowsInBatch = BATCH_SIZE;
                }
                
                if (batchIdx > 0) last_table_scanned_h->getTableBatch(batchIdx);
                
                unsigned int batchCount = 0;
                if (IS_GPU) {
                    batchCount = ExecuteCountString(columnIdx, last_table_scanned_h, rowsInBatch);
                }
                else {
                    batchCount = ExecuteCountStringCPU(columnIdx, last_table_scanned_h, rowsInBatch);
                }
                batchResults.push_back(batchCount);
            }
            
            unsigned int totalCount = 0;
            for (unsigned int count : batchResults) {
                totalCount += count;
            }

            WriteAggregationToCSV("count", columnName, &totalCount, columnType, "UINT", AGG_OUT_PATH);
        } 
        else if (columnType == "FLOAT") {
            int numBatches = last_table_scanned_h->getNumBatches();
            long long totalRows = last_table_scanned_h->getNumRows();
            std::vector<unsigned int> batchResults;
            
            for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
                long long rowsInBatch;
                if (batchIdx == numBatches - 1) {
                    rowsInBatch = totalRows % BATCH_SIZE;
                    if (rowsInBatch == 0 && totalRows > 0) {
                        rowsInBatch = BATCH_SIZE;
                    }
                } else {
                    rowsInBatch = BATCH_SIZE;
                }
                
                if (batchIdx > 0) last_table_scanned_h->getTableBatch(batchIdx);
                
                unsigned int batchCount = 0;
                if (IS_GPU) {
                    batchCount = ExecuteCountFloat(columnIdx, last_table_scanned_h, rowsInBatch);
                }
                else {
                    batchCount = ExecuteCountFloatCPU(columnIdx, last_table_scanned_h, rowsInBatch);
                }
                batchResults.push_back(batchCount);
            }
            
            unsigned int totalCount = 0;
            for (unsigned int count : batchResults) {
                totalCount += count;
            }
            
            WriteAggregationToCSV("count", columnName, &totalCount, columnType, "UINT", AGG_OUT_PATH);
        } 
        else {
            std::cerr << "Error: COUNT operation not supported for column type " << columnType << std::endl;
        }
    } 
    else if (function == "avg"){
        if (columnType == "FLOAT"){
            
            int numBatches = last_table_scanned_h->getNumBatches();
            long long totalRows = last_table_scanned_h->getNumRows();
            
            std::vector<double> batchSums;
            std::vector<unsigned int> batchCounts;
            
            for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
                long long rowsInBatch;
                if (batchIdx == numBatches - 1) {
                    rowsInBatch = totalRows % BATCH_SIZE;
                    if (rowsInBatch == 0 && totalRows > 0) {
                        rowsInBatch = BATCH_SIZE;
                    }
                } else {
                    rowsInBatch = BATCH_SIZE;
                }
                
                if (batchIdx > 0) 
                    last_table_scanned_h->getTableBatch(batchIdx);
                
                double batchSum = 0;
                unsigned int batchCount = 0;
                if (IS_GPU) {
                    batchSum = ExecuteSumFloat(columnIdx, last_table_scanned_h, rowsInBatch);
                    batchCount = ExecuteCountFloat(columnIdx, last_table_scanned_h, rowsInBatch);
                }
                else {
                    batchSum = ExecuteSumFloatCPU(columnIdx, last_table_scanned_h, rowsInBatch);
                    batchCount = ExecuteCountFloat(columnIdx, last_table_scanned_h, rowsInBatch);
                }
                
                batchSums.push_back(batchSum);
                batchCounts.push_back(batchCount);
            }
            
            double finalSum = 0.0;
            unsigned int totalCount = 0;
            
            for (int i = 0; i < batchSums.size(); i++) {
                finalSum += batchSums[i];
                totalCount += batchCounts[i];
            }
            
            // Calculate average
            double average = 0.0;
            if (totalCount > 0) {
                average = finalSum / totalCount;
            }
            
            WriteAggregationToCSV("avg", columnName, &average, "FLOAT", "DOUBLE", AGG_OUT_PATH);
        }
        else {
             std::cerr << "Error: AVG operation not supported for column type " << columnType << std::endl;
        }
    }
    
    else {
        std::cerr << "Error: Unsupported aggregate function: " << function << std::endl;
    }
}

void ExecuteJoin(Kernel_Condition** conditions, int num_total_conditions, const std::vector<int>& conditions_groups_sizes, const std::vector<std::string>& projections, Table* last_table_scanned_1, Table* last_table_scanned_2, int*& h_out1, int*& h_out2, int& h_result_count) {  
    int num_groups_conditions = conditions_groups_sizes.size();
    void** h_data1 = last_table_scanned_1->getData();
    void** h_data2 = last_table_scanned_2->getData();
    char** h_names1 = last_table_scanned_1->getColumnNames();
    char** h_names2 = last_table_scanned_2->getColumnNames();

    int rows1 = last_table_scanned_1->getNumRows();
    int rows2 = last_table_scanned_2->getNumRows();
    int cols1 = last_table_scanned_1->getNumColumns();
    int cols2 = last_table_scanned_2->getNumColumns();
    int totalPairs = rows1 * rows2;
    h_out1 = new int[totalPairs];
    h_out2 = new int[totalPairs];

    // std::cout << "Table 1: " << rows1 << " rows\n";
    // std::cout << "Table 2: " << rows2 << " rows\n";
    // std::cout << "Total join comparisons: " << totalPairs << "\n";

    void **d_in1 = nullptr, **d_in2 = nullptr;
    char **d_in1_names = nullptr, **d_in2_names = nullptr;
    
    Kernel_Condition* flat_conditions = new Kernel_Condition[num_total_conditions];
    int* group_offsets = new int[num_groups_conditions]; // where each group starts
    int* group_sizes = new int[num_groups_conditions];   // size of each group

    int *d_out1 = nullptr, *d_out2 = nullptr;
    int *d_result_count = nullptr;

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (totalPairs + threadsPerBlock - 1) / threadsPerBlock;

    Kernel_Condition* d_flat_conditions;
    int* d_group_offsets;
    int* d_group_sizes;

    int flat_index = 0;
    for (int g = 0; g < num_groups_conditions; ++g) {
        group_offsets[g] = flat_index;
        int group_size = 0;
        for (int i = 0; i<conditions_groups_sizes[g]; ++i) { 
            flat_conditions[flat_index++] = conditions[g][i];
            ++group_size;
        }

        group_sizes[g] = group_size;
    }

    cudaMalloc((void **)&d_in1, cols1 * sizeof(void*));
    cudaMalloc((void **)&d_in2, cols2 * sizeof(void*));
    cudaMalloc((void **)&d_in1_names, cols1 * sizeof(char*));
    cudaMalloc((void **)&d_in2_names, cols2 * sizeof(char*));
    cudaMalloc(&d_flat_conditions, num_total_conditions * sizeof(Kernel_Condition));
    cudaMalloc(&d_group_offsets, num_groups_conditions * sizeof(int));
    cudaMalloc(&d_group_sizes, num_groups_conditions * sizeof(int));

    cudaMalloc((void **)&d_out1, totalPairs * sizeof(int));
    cudaMalloc((void **)&d_out2, totalPairs * sizeof(int));
    cudaMalloc((void **)&d_result_count, sizeof(int));
    cudaMemset(d_result_count, 0, sizeof(int));

    cudaMemcpy(d_in1, h_data1, cols1 * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, h_data2, cols2 * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in1_names, h_names1, cols1 * sizeof(char*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2_names, h_names2, cols2 * sizeof(char*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flat_conditions, flat_conditions, num_total_conditions * sizeof(Kernel_Condition), cudaMemcpyHostToDevice);
    cudaMemcpy(d_group_offsets, group_offsets, num_groups_conditions * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_group_sizes, group_sizes, num_groups_conditions * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start, 0);

    join<<<blocksPerGrid, threadsPerBlock>>>(
        d_in1, d_in2, d_in1_names, d_in2_names,
        d_out1, d_out2,
        d_flat_conditions, d_group_offsets, d_group_sizes,
        num_groups_conditions, rows1, rows2, d_result_count);    

    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);

    // Copy results count
    h_result_count = 0;
    cudaMemcpy(&h_result_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost);
    //std::cout << "Join completed. Matches found: " << h_result_count << ", GPU Time: " << gpu_time << " ms\n";

    // Copy back matched indices
    cudaMemcpy(h_out1, d_out1, h_result_count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out2, d_out2, h_result_count * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_in1_names);
    cudaFree(d_in2_names);
    cudaFree(d_flat_conditions);
    cudaFree(d_group_offsets);
    cudaFree(d_group_sizes);
    cudaFree(d_out1);
    cudaFree(d_out2);
    cudaFree(d_result_count);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
}