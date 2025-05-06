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

// Function to trim whitespace
std::string trim(const std::string& str) {
    auto start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    auto end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

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
            std::regex condition_pattern(R"((\S+)\s*(=|!=|<|>|>=|<=)\s*(\S+))");
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

float ExecuteMinMaxFloat(int columnIdx, bool findMin, Table* last_table_scanned_h) {
    // Get the data for the specified column
    void* columnData = last_table_scanned_h->getData()[columnIdx];
    float* h_data = static_cast<float*>(columnData);
    size_t numRows = BATCH_SIZE;
    
    // ================= CPU Implementation =================
    cudaEvent_t cpu_start, cpu_stop;
    cudaEventCreate(&cpu_start);
    cudaEventCreate(&cpu_stop);
    float cpu_time = 0.0f;
    
    // Start CPU timing
    cudaEventRecord(cpu_start, 0);
    
    float cpu_result = findMin ? FLT_MAX : -FLT_MAX;
    
    // Compute min/max on CPU
    for (size_t i = 0; i < numRows; i++) {
        if (findMin) {
            cpu_result = fminf(cpu_result, h_data[i]);
        } else {
            cpu_result = fmaxf(cpu_result, h_data[i]);
        }
    }
    
    // Stop CPU timing
    cudaEventRecord(cpu_stop, 0);
    cudaEventSynchronize(cpu_stop);
    cudaEventElapsedTime(&cpu_time, cpu_start, cpu_stop);
    
    // ================= GPU Implementation =================
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    float gpu_time = 0.0f;
    
    float *d_data = nullptr;
    float *d_block_results = nullptr;
    
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (numRows + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate memory on the GPU
    cudaMalloc((void **)&d_data, numRows * sizeof(float));
    cudaMalloc((void **)&d_block_results, blocksPerGrid * sizeof(float));
    
    // Copy data from host to device 
    cudaMemcpy(d_data, h_data, numRows * sizeof(float), cudaMemcpyHostToDevice);
    
    // Start GPU timing
    cudaEventRecord(gpu_start, 0);
    
    // Find min/max for each block using the appropriate kernel
    if (findMin) {
        findWarpMin<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_block_results, numRows);
    } else {
        findWarpMax<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_block_results, numRows);
    }
    
    // Copy block results back to host for CPU final reduction
    float *h_block_results = new float[blocksPerGrid];
    cudaMemcpy(h_block_results, d_block_results, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    
    // CPU performs final reduction on block results
    float gpu_result = findMin ? FLT_MAX : -FLT_MAX;
    for (int i = 0; i < blocksPerGrid; i++) {
        if (findMin) {
            gpu_result = fminf(gpu_result, h_block_results[i]);
        } else {
            gpu_result = fmaxf(gpu_result, h_block_results[i]);
        }
    }
    
    // Stop GPU timing
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "\nCUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Clean up
    cudaFree(d_data);
    cudaFree(d_block_results);
    delete[] h_block_results;
    cudaEventDestroy(cpu_start);
    cudaEventDestroy(cpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    
    return gpu_result;
}

long long ExecuteMinMaxDate(int columnIdx, bool findMin, Table* last_table_scanned_h) {
    // Get the data for the specified column
    void* columnData = last_table_scanned_h->getData()[columnIdx];
    char** dateStrings = static_cast<char**>(columnData);
    size_t numRows = BATCH_SIZE;
    
    if (numRows == 0) {
        std::cout << "No data to process for " << (findMin ? "MIN" : "MAX") << " DATE operation" << std::endl;
        return findMin ? LLONG_MAX : LLONG_MIN;
    }
    
    // ================= GPU Implementation =================
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    float gpu_time = 0.0f;
    
    // Create a contiguous array of date strings for GPU processing
    char* h_contiguous_strings = (char*)malloc(numRows * MAX_DATETIME * sizeof(char));
    for (size_t i = 0; i < numRows; i++) {
        strncpy(h_contiguous_strings + i * MAX_DATETIME, dateStrings[i], MAX_DATETIME - 1);
        h_contiguous_strings[i * MAX_DATETIME + MAX_DATETIME - 1] = '\0'; // Ensure null termination
    }
    
    // Allocate device memory
    char* d_date_strings = nullptr;
    long long* d_dates = nullptr;
    long long* d_block_results = nullptr;
    
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (numRows + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_date_strings, numRows * MAX_DATETIME * sizeof(char));
    cudaMalloc((void**)&d_dates, numRows * sizeof(long long));
    cudaMalloc((void**)&d_block_results, blocksPerGrid * sizeof(long long));
    
    // Copy date strings to device
    cudaMemcpy(d_date_strings, h_contiguous_strings, numRows * MAX_DATETIME * sizeof(char), cudaMemcpyHostToDevice);
    
    // Start GPU timing
    cudaEventRecord(gpu_start, 0);
    
    // Convert date strings to int64 representation on GPU
    convertDatesToInt64Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_date_strings, d_dates, numRows, MAX_DATETIME, findMin);
    
    // Find min/max date for each block using the warp-optimized kernel
    if (findMin) {
        findWarpMinDate<<<blocksPerGrid, threadsPerBlock>>>(d_dates, d_block_results, numRows);
    } else {
        findWarpMaxDate<<<blocksPerGrid, threadsPerBlock>>>(d_dates, d_block_results, numRows);
    }
    
    // Copy block results back to host for CPU final reduction
    long long* h_block_results = new long long[blocksPerGrid];
    cudaMemcpy(h_block_results, d_block_results, blocksPerGrid * sizeof(long long), cudaMemcpyDeviceToHost);
    
    // CPU performs final reduction on block results
    long long gpu_result = findMin ? LLONG_MAX : LLONG_MIN;
    for (int i = 0; i < blocksPerGrid; i++) {
        long long block_val = h_block_results[i];
        
        if (block_val != (findMin ? LLONG_MAX : LLONG_MIN) && block_val > 0) {
            if ((findMin && (gpu_result == LLONG_MAX || block_val < gpu_result)) ||
                (!findMin && block_val > gpu_result)) {
                gpu_result = block_val;
            }
        }
    }
    
    // Stop GPU timing
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "\nCUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Clean up
    free(h_contiguous_strings);
    cudaFree(d_date_strings);
    cudaFree(d_dates);
    cudaFree(d_block_results);
    delete[] h_block_results;
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    
    return gpu_result;
}

double ExecuteSumFloat(int columnIdx, Table* last_table_scanned_h) {
    // Get the data for the specified column
    void* columnData = last_table_scanned_h->getData()[columnIdx];
    float* h_data = static_cast<float*>(columnData);
    size_t numRows = BATCH_SIZE;
    
    if (numRows == 0) {
        std::cout << "No data to process for SUM operation" << std::endl;
        return 0;
    }
    
    // ================= GPU Implementation =================
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    float gpu_time = 0.0f;
    
    // Allocate device memory
    float* d_data = nullptr;
    double* d_sum = nullptr;
    
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (numRows + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_data, numRows * sizeof(float));
    cudaMalloc((void**)&d_sum, sizeof(double));
    
    // Copy data from host to device (not included in timing)
    cudaMemcpy(d_data, h_data, numRows * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize sum to 0
    double zero = 0.0;
    cudaMemcpy(d_sum, &zero, sizeof(double), cudaMemcpyHostToDevice);
    
    // Start GPU timing
    cudaEventRecord(gpu_start, 0);
    
    // Launch kernel with atomic adds
    sumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_sum, numRows);
    
    // Stop GPU timing
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
    
    // Copy the result back to the host
    double gpu_sum;
    cudaMemcpy(&gpu_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "\nCUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Clean up
    cudaFree(d_data);
    cudaFree(d_sum);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    return gpu_sum;
}

unsigned int ExecuteCountString(int columnIdx, Table* last_table_scanned_h) {
    // Get the data for the specified column
    void* columnData = last_table_scanned_h->getData()[columnIdx];
    char** stringData = static_cast<char**>(columnData);
    size_t numRows = BATCH_SIZE;
    
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

unsigned int ExecuteCountFloat(int columnIdx, Table* last_table_scanned_h) {
    // Get the data for the specified column
    void* columnData = last_table_scanned_h->getData()[columnIdx];
    float* floatData = static_cast<float*>(columnData);
    size_t numRows = BATCH_SIZE;
    
    if (numRows == 0) {
        std::cout << "No data to process for COUNT operation" << std::endl;
        return 0;
    }
    
    // ================= GPU Implementation =================
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    float gpu_time = 0.0f;
    
    // Allocate device memory
    float* d_floats = nullptr;
    unsigned int* d_count = nullptr;
    
    // Calculate kernel launch parameters
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (numRows + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_floats, numRows * sizeof(float));
    cudaMalloc((void**)&d_count, sizeof(unsigned int));
    
    // Copy data from host to device
    cudaMemcpy(d_floats, floatData, numRows * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize count to 0
    unsigned int zero = 0;
    cudaMemcpy(d_count, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    // Start GPU timing
    cudaEventRecord(gpu_start, 0);
    
    // Launch kernel
    countNonNullFloatsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_floats, numRows, d_count);
    
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
    cudaFree(d_floats);
    cudaFree(d_count);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    
    return gpu_count;
}


void ExecuteAggregateFunction(const std::string& function, int columnIdx, Table* last_table_scanned_h) {
    if (columnIdx >= last_table_scanned_h->getNumColumns()) {
        std::cerr << "Error: Column index " << columnIdx << " out of bounds" << std::endl;
        return;
    }
    
    // Get column name and type
    std::string columnName = last_table_scanned_h->getColumnNames()[columnIdx];
    std::string columnType;
    
    // Extract column type from column name
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
    
    std::cout << "Executing " << function << " on column " << columnName 
              << " (type: " << columnType << ", index: " << columnIdx << ")" << std::endl;
    
    // Route to the appropriate kernel function based on function name and data type
    if (function == "max") {
        if (columnType == "FLOAT") {
            float maxValue = ExecuteMinMaxFloat(columnIdx, false, last_table_scanned_h); // find_min = false
            std::cout<<"Max Value: "<< maxValue << std::endl;
        } else if (columnType == "DATE") {
            long long res = ExecuteMinMaxDate(columnIdx, false, last_table_scanned_h);
            char date[MAX_DATETIME];
            int64ToDate(res,date);
            std::cout<<"Max Date: "<< date << std::endl;
        } else {
            std::cerr << "Error: MAX operation not supported for column type " << columnType << std::endl;
        }
    } else if (function == "min") {
        if (columnType == "FLOAT") {
            float minValue = ExecuteMinMaxFloat(columnIdx, true, last_table_scanned_h); // find_min = true
            std::cout<<"Min Value: "<< minValue << std::endl;
        } else if (columnType == "DATE") {
            long long res = ExecuteMinMaxDate(columnIdx, true, last_table_scanned_h);
            char date[MAX_DATETIME];
            int64ToDate(res,date);
            std::cout<<"Min Date: "<< date << std::endl;
        } else {
            std::cerr << "Error: MIN operation not supported for column type " << columnType << std::endl;
        }
    } else if (function == "sum") {
        if (columnType == "FLOAT") {
            double res = ExecuteSumFloat(columnIdx, last_table_scanned_h);
            std::cout<<"Sum Result: "<< res << std::endl;
        } else {
            std::cerr << "Error: SUM operation not supported for column type " << columnType << std::endl;
        }
    } 
    else if (function == "count") {
    if (columnType == "TEXT" || columnType == "DATE") {
        unsigned int res = ExecuteCountString(columnIdx, last_table_scanned_h);
        std::cout<<"Count Result: "<< res << std::endl;
    } 
    else if (columnType == "FLOAT"){
        unsigned int res = ExecuteCountFloat(columnIdx, last_table_scanned_h);
        std::cout<<"Count Result: "<< res << std::endl;
    }
    else {
        std::cerr << "Error: COUNT operation not supported for column type " << columnType << std::endl;
    }
}
    else {
        std::cerr << "Error: Unsupported aggregate function: " << function << std::endl;
    }
}