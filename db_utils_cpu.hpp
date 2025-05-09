#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <float.h>
#include <limits.h>
#include <cmath>
#include "Table.hpp"
#include "config.hpp"
#include "utils.hpp"

float ExecuteMinMaxFloatCPU(int columnIdx, bool findMin, Table* last_table_scanned_h, long long numRows);
long long ExecuteMinMaxDateCPU(int columnIdx, bool findMin, Table* last_table_scanned_h, long long numRows);
double ExecuteSumFloatCPU(int columnIdx, Table* last_table_scanned_h, long long numRows);
unsigned int ExecuteCountStringCPU(int columnIdx, Table* last_table_scanned_h, long long numRows);
unsigned int ExecuteCountFloatCPU(int columnIdx, Table* last_table_scanned_h, long long numRows);
bool ExecuteSortBatchCPU(int columnIdx, bool isAscending, Table* table, long long rowsInBatch);
void ExecuteJoinCPU(Kernel_Condition** conditions, int num_total_conditions, 
                 const std::vector<int>& conditions_groups_sizes, 
                 const std::vector<std::string>& projections, 
                 Table* last_table_scanned_1, Table* last_table_scanned_2, 
                 int*& h_out1, int*& h_out2, int& h_result_count);

