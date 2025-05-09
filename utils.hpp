#pragma once

#include <string>
#include <vector>

struct Condition {
    std::string left_operand;
    std::string relational_operator;
    std::string right_operand;
};

struct Kernel_Condition {
    char relational_operator;
    char type;
    int idx1;
    int idx2;
    void* value = nullptr;
};

// Function to trim whitespace
std::string trim(const std::string& str);