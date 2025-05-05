#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include "utils.hpp"

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