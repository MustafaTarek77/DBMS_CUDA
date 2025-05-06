#pragma once

#include <string>
#include <vector>
#include "Table.hpp"

std::string trim(const std::string& str);
std::string removeOuterParentheses(const std::string& expr);
std::vector<std::string> splitByWord(const std::string& str, const std::string& word);
std::vector<std::string> splitByOperator(const std::string& str, const std::string& op);
void parseComplexExpression(const std::string& expr, std::vector<std::vector<Condition>>& conditions);
void printConditionStructure(const std::vector<std::vector<Condition>>& conditions);
void ExecuteAggregateFunction(const std::string& function, int columnIdx, Table* last_table_scanned_h);