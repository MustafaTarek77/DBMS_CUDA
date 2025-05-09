#pragma once

#include <string>
#include <vector>
#include "Table.hpp"

std::string trim(const std::string& str);
std::string removeOuterParentheses(const std::string& expr);
std::vector<std::string> splitByWord(const std::string& str, const std::string& word);
bool ExecuteSortBatch(int columnIdx, bool isAscending, Table* table, long long rowsInBatch , int direction);
std::vector<std::string> splitByOperator(const std::string& str, const std::string& op);
void parseComplexExpression(const std::string& expr, std::vector<std::vector<Condition>>& conditions);
void printConditionStructure(const std::vector<std::vector<Condition>>& conditions);
void ExecuteAggregateFunction(const std::string& function, int columnIdx, Table* last_table_scanned, const std::string& AGG_OUT_PATH, bool IS_GPU);
void ExecuteJoin(Kernel_Condition** conditions, int total_conditions, const std::vector<int>& conditions_groups_sizes, const std::vector<std::string>& projections, Table* last_table_scanned_1, Table* last_table_scanned_2, int*& h_out1, int*& h_out2, int& h_result_count);
bool ExecuteSortBatch(int columnIdx, bool isAscending, Table* table, long long rowsInBatch);