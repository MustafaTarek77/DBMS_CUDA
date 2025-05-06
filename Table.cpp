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

bool Table::checkConditions(const std::vector<std::vector<Condition>>& conditions, const std::string& row) {
    if (conditions.empty()) {
        return true;
    }

    std::stringstream ss(row);
    std::string value;
    int col = 0;
    int idx = 0;
    std::unordered_map<std::string, std::string> rowData;

    // Step 1: Parse row values and store relevant ones in a map
    while (std::getline(ss, value, ',')) {
        value.erase(value.find_last_not_of(" \r\n\t") + 1);

        if (projection_indices[idx] == col) {
            std::string header = std::string(headers[idx]);
            std::string colName = header.substr(0, header.find("("));
            rowData[colName] = value;
            idx++;
        }

        if (idx == numHeaders)
            break;

        col++;
    }

    // Step 2: Evaluate each group (OR between groups)
    int groupIdx = 0;
    for (const auto& group : conditions) {
        bool groupResult = true;

        // Step 3: Evaluate each condition in the group (AND within group)
        for (const auto& cond : group) {
            auto it = rowData.find(cond.left_operand);
            if (it == rowData.end()) {
                groupResult = false;
                break;
            }

            const std::string& leftVal = it->second;
            std::string headerType;
            bool foundHeader = false;

            for (int i = 0; i < numHeaders; i++) {
                std::string colName = headers[i];
                std::string baseColName = colName.substr(0, colName.find("("));
                if (baseColName == cond.left_operand) {
                    headerType = colName;
                    foundHeader = true;
                    break;
                }
            }

            if (!foundHeader) {
                groupResult = false;
                break;
            }

            bool conditionResult = false;
            if (headerType.find("(N)") != std::string::npos) {
                conditionResult = compareFloats(std::stof(leftVal), cond.relational_operator, std::stof(cond.right_operand));
            }
            else if (headerType.find("(T)") != std::string::npos) {
                conditionResult = compareStrings(leftVal, cond.relational_operator, cond.right_operand);
            }
            else if (headerType.find("(D)") != std::string::npos) {
                conditionResult = compareDateTime(leftVal, cond.relational_operator, cond.right_operand);
            } else {
                groupResult = false;
                break;
            }

            if (!conditionResult) {
                groupResult = false;
                break;
            }
        }

        if (groupResult) {
            return true; // OR between groups — return if any group is satisfied
        } 
        groupIdx++;
    }

    return false;
}

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

    /****************** Getting Headers ******************/
    numHeaders = projections.size();
    headers = new char*[numHeaders];
    numColumns = target_columns.size();
    columnNames = new char*[numColumns];
    // Read header
    if (std::getline(file, line)) {
        std::string header;
        std::stringstream ssParse(line);
        int col = 0;
        int headers_idx = 0, columns_idx=0;

        while (std::getline(ssParse, header, ',')) {
            header.erase(header.find_last_not_of(" \r\n\t") + 1);
            for(int i=0; i<numHeaders; i++) {
                std::string temp = header.substr(0, header.find("("));
                if(projections[i] == temp){
                    headers[headers_idx] = new char[header.length() + 1];
                    std::strcpy(headers[headers_idx], header.c_str());
                    projection_indices.push_back(col);
                    headers_idx++;
                }
                if (i<numColumns && target_columns[i] == temp) {
                    columnNames[columns_idx] = new char[header.length() + 1];
                    std::strcpy(columnNames[columns_idx], header.c_str());
                    target_indices.push_back(col);
                    columns_idx++;
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
            data[i] = static_cast<void*>(colData);
            data_temp[i] = static_cast<void*>(colTemp);
        } 
        else if (header.find("(T)") != std::string::npos) {
            char** colData = new char*[BATCH_SIZE];
            char** colTemp = new char*[BATCH_SIZE];
            for (int j = 0; j < BATCH_SIZE; ++j) {
                colData[j] = new char[MAX_VAR_CHAR + 1](); // zero-initialized
                colTemp[j] = new char[MAX_VAR_CHAR + 1]();
            }
            data[i] = static_cast<void*>(colData);
            data_temp[i] = static_cast<void*>(colTemp);
        } 
        else if (header.find("(D)") != std::string::npos) {
            char** colData = new char*[BATCH_SIZE];
            char** colTemp = new char*[BATCH_SIZE];
            for (int j = 0; j < BATCH_SIZE; ++j) {
                colData[j] = new char[MAX_DATETIME + 1]();
                colTemp[j] = new char[MAX_DATETIME + 1]();
            }
            data[i] = static_cast<void*>(colData);
            data_temp[i] = static_cast<void*>(colTemp);
        }
    }

    /****************** Getting Data ******************/
    // Read rows
    int row = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        int col = 0;
        int idx = 0;

        if (!checkConditions(conditions, line)) {
            continue;
        }
        
        while (std::getline(ss, value, ',')) {
            value.erase(value.find_last_not_of(" \r\n\t") + 1);
            if(target_indices[idx] == col)
            {
                std::string header = std::string(columnNames[idx]);
                if (batch_idx == 0) {
                    if (header.find("(N)") != std::string::npos) {
                        float entry = std::stof(value); 
                        static_cast<float*>(data[idx])[row] = entry;
                    } 
                    else if (header.find("(T)") != std::string::npos) {
                        std::strncpy(static_cast<char**>(data[idx])[row], value.c_str(), MAX_VAR_CHAR);
                        static_cast<char**>(data[idx])[row][MAX_VAR_CHAR] = '\0';  // Ensure null termination
                    }
                    else if (header.find("(D)") != std::string::npos) {
                        std::strncpy(static_cast<char**>(data[idx])[row], value.c_str(), MAX_DATETIME);
                        static_cast<char**>(data[idx])[row][MAX_DATETIME] = '\0';  // Ensure null termination
                    }
                } else {
                    if (header.find("(N)") != std::string::npos) {
                        float entry = std::stof(value);  
                        static_cast<float*>(data_temp[idx])[row] = entry;
                    } 
                    else if (header.find("(T)") != std::string::npos) {
                        std::strncpy(static_cast<char**>(data_temp[idx])[row], value.c_str(), MAX_VAR_CHAR);
                        static_cast<char**>(data_temp[idx])[row][MAX_VAR_CHAR] = '\0';  // Ensure null termination
                    }
                    else if (header.find("(D)") != std::string::npos) {
                        std::strncpy(static_cast<char**>(data_temp[idx])[row], value.c_str(), MAX_DATETIME);
                        static_cast<char**>(data_temp[idx])[row][MAX_DATETIME] = '\0';  // Ensure null termination
                    }
                }                
                idx++;
            }

            if(idx==numColumns)
                break;

            col++;
        }
        row++;
        if (row == BATCH_SIZE ) {
            if (!makeBatchFile(data_temp, BATCH_SIZE))
                return false;

            numRows+=row;
            row = 0;
            batch_idx++;
        }
    }

    if (row != 0 ) {
        if (!makeBatchFile(data_temp, row))
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

bool Table::makeBatchFile(void** const &data_temp, const int& size) {
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
        for (int row = 0; row < size; ++row) {
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
        for (int row = 0; row < size; ++row) {
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
    return batch_idx;
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