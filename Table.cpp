#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem> 
#include <cstring>
#include <ctime>
#include <iomanip>
#include <cmath>
#include "Table.hpp"
#include "duckdb.hpp"

namespace fs = std::filesystem;

Table::Table(const std::string &table_name, const std::vector<std::string>& projections, const std::vector<std::string>& target_columns, const std::vector<std::vector<Condition>>& conditions) {
    this->table_name = table_name;
    if(makeTableBatches(projections, target_columns, conditions))
        std::cout<<"Table Batches are created successfully in the disk"<<std::endl;
    else
        std::cout<<"Error while creating table batches in the disk"<<std::endl;
}

bool Table::makeTableBatches(const std::vector<std::string>& projections, const std::vector<std::string>& target_columns, const std::vector<std::vector<Condition>>& conditions) {
    std::string dir_name = TABLE_PATH + table_name;
    if (!fs::exists(dir_name)) {
        fs::create_directory(dir_name);
    }
    for (const auto& entry : fs::directory_iterator(FOLDER_PATH)) {
        if (entry.path().extension() == ".csv") {
            std::string filename = fs::path(entry.path()).stem().string();
            if (filename == table_name) {
                return makeBatches(entry.path().string(), projections, target_columns, conditions);
            }
        }
    }
    return false;
} 

bool Table::makeBatches(const std::string& filepath, const std::vector<std::string>& projections, const std::vector<std::string>& target_columns, const std::vector<std::vector<Condition>>& conditions) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return false;
    }

    std::string line;

    int group_num = 1;
    for (const auto& group : conditions) {
        std::cout << "Group " << group_num++ << ":\n";
        for (const auto& cond : group) {
            std::cout << "  " << cond.left_operand << " "
                      << cond.relational_operator << " "
                      << cond.right_operand << "\n";
        }
    }

    /****************** Getting Headers ******************/
    numColumns = projections.size();
    columnNames = new char*[numColumns];
    // Read header
    if (std::getline(file, line)) {
        std::string header;
        std::stringstream ssParse(line);
        int col = 0;
        int idx = 0;

        while (std::getline(ssParse, header, ',')) {
            header.erase(header.find_last_not_of(" \r\n\t") + 1);
            for(int i=0; i<numColumns; i++) {
                std::string temp = header.substr(0, header.find("("));
                if(projections[i] == temp){
                    columnNames[idx] = new char[header.length() + 1];
                    std::strcpy(columnNames[idx], header.c_str());
                    target_indices.push_back(col);
                    idx++;
                }
            }
            ++col;
        }
    }

    /****************** Initializing Memory ******************/
    data = new void*[numColumns];
    void** data_temp = new void*[numColumns];
    for (int i = 0; i < numColumns; ++i) {
        std::string header = std::string(columnNames[i]);
        if (header.find("(N)") != std::string::npos) {
            float* colData = new float[BATCH_SIZE];
            float* colTemp = new float[BATCH_SIZE];
            data[i] = colData;
            data_temp[i] = colTemp;
        } 
        else if (header.find("(T)") != std::string::npos) {
            char** colData = new char*[BATCH_SIZE];
            char** colTemp = new char*[BATCH_SIZE];
            for (int j = 0; j < BATCH_SIZE; ++j) {
                colData[j] = new char[MAX_VAR_CHAR + 1](); // zero-initialized
                colTemp[j] = new char[MAX_VAR_CHAR + 1]();
            }
            data[i] = colData;
            data_temp[i] = colTemp;
        } 
        else if (header.find("(D)") != std::string::npos) {
            char** colData = new char*[BATCH_SIZE];
            char** colTemp = new char*[BATCH_SIZE];
            for (int j = 0; j < BATCH_SIZE; ++j) {
                colData[j] = new char[MAX_DATETIME + 1]();
                colTemp[j] = new char[MAX_DATETIME + 1]();
            }
            data[i] = colData;
            data_temp[i] = colTemp;
        }
    }

    /****************** Getting Data ******************/
    // Read rows
    int row = 0;
    int batch_idx = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        int col = 0;
        int idx = 0;
        bool is_filtered = false;
        //check conditions
        while (std::getline(ss, value, ',')) {
            value.erase(value.find_last_not_of(" \r\n\t") + 1);
            if(target_indices[idx] == col)
            {
                std::string header = std::string(columnNames[idx]);
                if (batch_idx == 0) {
                    if (header.find("(N)") != std::string::npos) {
                        float* colData = static_cast<float*>(data[idx]);
                        float entry = std::stof(value); 
                        colData[row] = entry; 
                        //std::string key = header.substr(0, header.find("("));            
                        // auto it = conditions.find(key);
                        // if (it != conditions.end()) {
                        //     const Condition& cond = it->second;
                        
                        //     std::string op = cond.relational_operator;
                        //     float rhs = std::stof(cond.right_operand);
                        
                        //     if (compareFloats(entry, op, rhs)) {
                        //         colData[row] = entry;
                        //     }
                        //     else {
                        //         is_filtered = true;
                        //     }
                        // } 
                        // else {
                        //     colData[row] = entry;
                        // }
                    } 
                    else if (header.find("(T)") != std::string::npos) {
                        char** colData = static_cast<char**>(data[idx]);
                        std::strncpy(colData[row], value.c_str(), MAX_VAR_CHAR);
                        colData[row][MAX_VAR_CHAR] = '\0';  // Ensure null termination
                        // std::string key = header.substr(0, header.find("("));
                        // auto it = conditions.find(key);
                        // if (it != conditions.end()) {
                        //     const Condition& cond = it->second;
                        
                        //     std::string op = cond.relational_operator;
                        //     std::string rhs = cond.right_operand;
                        
                        //     if (compareStrings(value, op, rhs)) {
                        //         std::strncpy(colData[row], value.c_str(), MAX_VAR_CHAR);
                        //         colData[row][MAX_VAR_CHAR] = '\0';  // Ensure null termination
                        //     }
                        //     else {
                        //         is_filtered = true;
                        //     }
                        // } 
                        // else {
                        //     std::strncpy(colData[row], value.c_str(), MAX_VAR_CHAR);
                        //     colData[row][MAX_VAR_CHAR] = '\0';  // Ensure null termination
                        // }
                    }
                    else if (header.find("(D)") != std::string::npos) {
                        char** colData = static_cast<char**>(data[idx]);
                        std::strncpy(colData[row], value.c_str(), MAX_DATETIME);
                        colData[row][MAX_VAR_CHAR] = '\0';  // Ensure null termination
                        //std::string key = header.substr(0, header.find("("));
                        // auto it = conditions.find(key);
                        // if (it != conditions.end()) {
                        //     const Condition& cond = it->second;
                        
                        //     std::string op = cond.relational_operator;
                        //     std::string rhs = cond.right_operand;
                        
                        //     if (compareDateTime(value, op, rhs)) {
                        //         std::strncpy(colData[row], value.c_str(), MAX_DATETIME);
                        //         colData[row][MAX_VAR_CHAR] = '\0';  // Ensure null termination
                        //     }
                        //     else {
                        //         is_filtered = true;
                        //     }
                        // } 
                        // else {
                        //     std::strncpy(colData[row], value.c_str(), MAX_DATETIME);
                        //     colData[row][MAX_VAR_CHAR] = '\0';  // Ensure null termination
                        // }
                    }
                } else {
                    if (header.find("(N)") != std::string::npos) {
                        float* colData = static_cast<float*>(data[idx]);
                        float entry = std::stof(value);  
                        colData[row] = entry;
                        //std::string key = header.substr(0, header.find("("));            
                        // auto it = conditions.find(key);
                        // if (it != conditions.end()) {
                        //     const Condition& cond = it->second;
                        
                        //     std::string op = cond.relational_operator;
                        //     float rhs = std::stof(cond.right_operand);
                        
                        //     if (compareFloats(entry, op, rhs)) {
                        //         colData[row] = entry;
                        //     }
                        //     else {
                        //         is_filtered = true;
                        //     }
                        // } 
                        // else {
                        //     colData[row] = entry;
                        // }
                    } 
                    else if (header.find("(T)") != std::string::npos) {
                        char** colData = static_cast<char**>(data[idx]);
                        std::strncpy(colData[row], value.c_str(), MAX_VAR_CHAR);
                        colData[row][MAX_VAR_CHAR] = '\0';  // Ensure null termination
                        // std::string key = header.substr(0, header.find("("));
                        // auto it = conditions.find(key);
                        // if (it != conditions.end()) {
                        //     const Condition& cond = it->second;
                        
                        //     std::string op = cond.relational_operator;
                        //     std::string rhs = cond.right_operand;
                        
                        //     if (compareStrings(value, op, rhs)) {
                        //         std::strncpy(colData[row], value.c_str(), MAX_VAR_CHAR);
                        //         colData[row][MAX_VAR_CHAR] = '\0';  // Ensure null termination
                        //     }
                        //     else {
                        //         is_filtered = true;
                        //     }
                        // } 
                        // else {
                        //     std::strncpy(colData[row], value.c_str(), MAX_VAR_CHAR);
                        //     colData[row][MAX_VAR_CHAR] = '\0';  // Ensure null termination
                        // }
                    }
                    else if (header.find("(D)") != std::string::npos) {
                        char** colData = static_cast<char**>(data[idx]);
                        std::strncpy(colData[row], value.c_str(), MAX_DATETIME);
                        colData[row][MAX_VAR_CHAR] = '\0';  // Ensure null termination
                        // std::string key = header.substr(0, header.find("("));
                        // auto it = conditions.find(key);
                        // if (it != conditions.end()) {
                        //     const Condition& cond = it->second;
                        
                        //     std::string op = cond.relational_operator;
                        //     std::string rhs = cond.right_operand;
                        
                        //     if (compareDateTime(value, op, rhs)) {
                        //         std::strncpy(colData[row], value.c_str(), MAX_DATETIME);
                        //         colData[row][MAX_VAR_CHAR] = '\0';  // Ensure null termination
                        //     }
                        //     else {
                        //         is_filtered = true;
                        //     }
                        // } 
                        // else {
                        //     std::strncpy(colData[row], value.c_str(), MAX_DATETIME);
                        //     colData[row][MAX_VAR_CHAR] = '\0';  // Ensure null termination
                        // }
                    }
                }                
                idx++;
            }

            if(idx==numColumns)
                break;

            col++;
        }
        if (!is_filtered) {
            row++;
        }
        if (row == BATCH_SIZE ) {
            if (!makeBatchFile(batch_idx, data_temp, BATCH_SIZE))
                return false;

            numRows+=row;
            row = 0;
            batch_idx++;
        }
    }

    if (row != 0 ) {
        if (!makeBatchFile(batch_idx, data_temp, row))
            return false;
        
        numRows+=row;
        row = 0;
        batch_idx++;
    }

    if (data_temp) {
        for (int i = 0; i < numColumns; ++i) {
            std::string header = std::string(columnNames[i]);
        
            if (header.find("(N)") != std::string::npos) {
                delete[] static_cast<float*>(data_temp[i]);
            } 
            else if (header.find("(T)") != std::string::npos) {
                char** colTemp = static_cast<char**>(data_temp[i]);
                for (int j = 0; j < BATCH_SIZE; ++j)
                    delete[] colTemp[j];
                delete[] colTemp;
            } 
            else if (header.find("(D)") != std::string::npos) {
                char** colTemp = static_cast<char**>(data_temp[i]);
                for (int j = 0; j < BATCH_SIZE; ++j)
                    delete[] colTemp[j];
                delete[] colTemp;
            }
        }
        delete[] data_temp;
        data_temp = nullptr;  
    }

    file.close();

    return true;
}  

bool Table::makeBatchFile(const int& batch_idx, void** const &data_temp, const int& size) {
    std::string folder_path = TABLE_PATH + table_name;
    std::string file_path = folder_path + "/BATCH" + std::to_string(batch_idx) + ".csv";

    // Make sure folder exists
    if (!fs::exists(folder_path)) {
        fs::create_directories(folder_path);  
    }

    // Then open the file
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "MAKE BATCH: Failed to open file "<< file_path << std::endl;
        return false;
    }
    
    if (batch_idx==0) {
        for (int row = 0; row < BATCH_SIZE; ++row) {
            for (int col = 0; col < numColumns; ++col) {
                std::string header = std::string(columnNames[col]);
                if (header.find("(N)") != std::string::npos) {
                    file << std::fixed << std::setprecision(FLOAT_PRECISION) << ((float*)data[col])[row];
                } 
                else if (header.find("(T)") != std::string::npos || header.find("(D)") != std::string::npos) {
                    file << ((char**)data[col])[row];
                }
                if (col != numColumns - 1)
                    file << ",";
            }
            file << "\n";
        }  
    }
    else {
        for (int row = 0; row < BATCH_SIZE; ++row) {
            for (int col = 0; col < numColumns; ++col) {
                std::string header = std::string(columnNames[col]);
        
                if (header.find("(N)") != std::string::npos) {
                    file << std::fixed << std::setprecision(FLOAT_PRECISION) << ((float*)data_temp[col])[row];
                } 
                else if (header.find("(T)") != std::string::npos || header.find("(D)") != std::string::npos) {
                    file << ((char**)data_temp[col])[row];
                }
                if (col != numColumns - 1)
                    file << ",";
            }
            file << "\n";
        }  
    }
    return true;
}

bool Table::getTableBatch(const int& batch_idx) {
    std::string folder_path = TABLE_PATH + table_name;
    std::string file_path = folder_path + "/BATCH" + std::to_string(batch_idx) + ".csv";

    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "GET BATCH: Failed to open file " << file_path << std::endl;
        return false;
    }

    std::string line;
    int row = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        int col = 0;
        while (std::getline(ss, value, ',')) {
            value.erase(value.find_last_not_of(" \r\n\t") + 1); 
            std::string header = std::string(columnNames[col]);
            if (header.find("(N)") != std::string::npos) {
                ((float*)data[col])[row] = std::stof(value);
            } 
            else if (header.find("(T)") != std::string::npos) {
                std::strncpy(((char**)data[col])[row], value.c_str(), MAX_VAR_CHAR); 
                ((char**)data[col])[row][MAX_VAR_CHAR] = '\0'; 
            }
            else if(header.find("(D)") != std::string::npos) {
                std::strncpy(((char**)data[col])[row], value.c_str(), MAX_DATETIME); 
                ((char**)data[col])[row][MAX_DATETIME] = '\0';
            }

            col++;
        }
        row++;
    }

    file.close();
    return true;
}

bool Table::compareFloats(float lhs, const std::string& op, float rhs) {
    if (op == "=") return lhs == rhs;
    if (op == "!=") return lhs != rhs;
    if (op == "<") return lhs < rhs;
    if (op == ">") return lhs > rhs;
    return false;
}

bool Table::compareStrings(const std::string& lhs, const std::string& op, const std::string& rhs) {
    if (op == "=") return lhs == rhs;
    if (op == "!=") return lhs != rhs;
    if (op == "<") return lhs < rhs;
    if (op == ">") return lhs > rhs;
    return false;
}

bool Table::compareDateTime(const std::string& lhs, const std::string& op, const std::string& rhs) {
    std::tm lhs_tm = {}, rhs_tm = {};
    std::istringstream lhs_ss(lhs), rhs_ss(rhs);

    lhs_ss >> std::get_time(&lhs_tm, "%Y-%m-%d %H:%M:%S");
    rhs_ss >> std::get_time(&rhs_tm, "%Y-%m-%d %H:%M:%S");

    if (lhs_ss.fail() || rhs_ss.fail()) {
        std::cerr << "Invalid timestamp format." << std::endl;
        return false;
    }

    std::time_t lhs_time = std::mktime(&lhs_tm);
    std::time_t rhs_time = std::mktime(&rhs_tm);

    if (op == "=") return lhs_time == rhs_time;
    if (op == "!=") return lhs_time != rhs_time;
    if (op == "<") return lhs_time < rhs_time;
    if (op == ">") return lhs_time > rhs_time;
    
    std::cerr << "Unsupported operator for timestamps: " << op << std::endl;
    return false;
}

int Table::getNumColumns() {
    return numColumns;
}

long long Table::getNumRows() {
    return numRows;
}

int Table::getNumBatches() {
    return std::ceil(numRows/BATCH_SIZE);
}

char** Table::getColumnNames() {
    return columnNames;
}

void** Table::getData() {
    return data;
}

void Table::printData() {
    if (!data || numColumns == 0 || numRows == 0) {
        std::cout << "No data to print.\n";
        return;
    }
    std::cout << "--------- Printing Data for Table: " << table_name << " ---------" << std::endl;
    // Print header
    for (int col = 0; col < numColumns; ++col) {
        std::cout << columnNames[col];
        if (col < numColumns - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    // Print rows
    for (int row = 0; row < HEAD && row < numRows; ++row) {
        for (int col = 0; col < numColumns; ++col) {
            std::string header = columnNames[col];
            if (header.find("(N)") != std::string::npos) {
                std::cout << ((float*)data[col])[row];
            } 
            else if (header.find("(T)") != std::string::npos || header.find("(D)") != std::string::npos) {
                std::cout << ((char**)data[col])[row];
            } 
            else {
                std::cout << "NULL";
            }

            if (col < numColumns - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
}


// Table::~Table() {
//     // Delete data
//     if (data) {
//         for (int i = 0; i < numColumns; ++i) {
//             std::string header = std::string(columnNames[i]);
        
//             if (header.find("(N)") != std::string::npos) {
//                 delete[] static_cast<float*>(data[i]);
//             } 
//             else if (header.find("(T)") != std::string::npos) {
//                 char** colTemp = static_cast<char**>(data[i]);
//                 for (int j = 0; j < BATCH_SIZE; ++j)
//                     delete[] colTemp[j];
//                 delete[] colTemp;
//             } 
//             else if (header.find("(D)") != std::string::npos) {
//                 char** colTemp = static_cast<char**>(data[i]);
//                 for (int j = 0; j < BATCH_SIZE; ++j)
//                     delete[] colTemp[j];
//                 delete[] colTemp;
//             }
//         }
//         delete[] data;
//         data = nullptr;  
//     }

//     // Delete column names
//     if (columnNames) {
//         for (int col = 0; col < numColumns; ++col) {
//             delete[] columnNames[col];
//         }
//         delete[] columnNames;
//         columnNames = nullptr;
//     }

//     numColumns = 0;
//     numRows = 0;
// }



// bool Table::readCSVColumns_DUCK(const std::string& filepath, const std::vector<std::string>& projections) {
//     // Step 1: Allocate memory for the data buffer
//     if (numRows == 0) {
//         numColumns = projections.size();
//         columnNames = new char*[numColumns];
        
//         for(int i=0; i<numColumns; i++) {
//             columnNames[i] = new char[projections[i].length() + 1];
//             std::strcpy(columnNames[i], projections[i].c_str());
//         }

//         data = new char**[numColumns];
//         for (int i = 0; i < numColumns; ++i) {
//             data[i] = new char*[BATCH_SIZE];
//             for (int j = 0; j < BATCH_SIZE; ++j) {
//                 data[i][j] = nullptr;
//             }
//         }

//     } else {
//         for (int i = 0; i < numColumns; ++i) {
//             for (int j = 0; j < BATCH_SIZE; ++j) {
//                 data[i][j] = nullptr;
//             }
//         }
//     }

//     // Step 2: Set up in-memory DuckDB and build query
//     duckdb::DuckDB db(nullptr); // in-memory DB
//     duckdb::Connection con(db);

//     std::string select_clause = "SELECT ";
//     for (size_t i = 0; i < numColumns; ++i) {
//         select_clause += "\"" + projections[i] + "\""; // handle spaces/special chars
//         if (i < numColumns - 1) {
//             select_clause += ", ";
//         }
//     }
//     std::string query = select_clause + " FROM '" + filepath + "'";

//     auto result = con.Query(query);
//     //std::cout<<query<<std::endl;
//     if (!result || result->HasError()) {
//         std::cerr << "Query error: " << result->GetError() << std::endl;
//         return false;
//     }

//     // Step 3: Read and store results
//     size_t row_index = 0;
//     while (auto chunk = result->Fetch()) {
//         for (size_t i = 0; i < chunk->size(); i++) {
//             for (size_t col = 0; col < numColumns; ++col) {
//                 std::string val = chunk->GetValue(col, i).ToString();
//                 data[col][row_index] = new char[val.length() + 1];
//                 std::strcpy(data[col][row_index], val.c_str());
//             }
//             row_index++;
//             if (row_index >= BATCH_SIZE) break;
//         }
//         if (row_index >= BATCH_SIZE) break;
//     }

//     numRows += row_index;

//     return true;
// }s