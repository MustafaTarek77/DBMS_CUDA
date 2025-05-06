#pragma once

#include <string>
#include <vector>

struct Condition {
    std::string left_operand;
    std::string relational_operator;
    std::string right_operand;
};

struct LastTableScanned {
    void** data = nullptr;
    char** columnNames = nullptr;
    std::vector<std::string> projections;
    std::vector<std::vector<Condition>> conditions;
    std::string table_name;
    int numColumns = 0;
    long long numRows = 0;
    int numBatches = 0;
};