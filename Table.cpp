#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem> 
#include <cstring>
#include "Table.hpp"
#include "duckdb.hpp"
namespace fs = std::filesystem;

Table::Table(const std::string &table_name, const std::vector<std::string>& projections) {
    this->table_name = table_name;
    if(makeTableBatches(projections))
        std::cout<<"Table Batches are created successfully in the disk"<<std::endl;
    else
        std::cout<<"Error while creating table batches in the disk"<<std::endl;
}

bool Table::makeTableBatches(const std::vector<std::string>& projections) {
    std::string dir_name = TABLE_PATH + table_name;
    if (!fs::exists(dir_name)) {
        fs::create_directory(dir_name);
    }
    for (const auto& entry : fs::directory_iterator(FOLDER_PATH)) {
        if (entry.path().extension() == ".csv") {
            std::string filename = fs::path(entry.path()).stem().string();
            if (filename == table_name) {
                return makeBatches(entry.path().string(), projections);
            }
        }
    }
    return false;
} 


bool Table::makeBatches(const std::string& filepath, const std::vector<std::string>& projections) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return false;
    }

    std::string line;

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
            for(int i=0; i<projections.size(); i++) {
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

    data = new char**[numColumns];
    char*** data_temp = new char**[numColumns];
    for (int i = 0; i < numColumns; ++i) {
        data[i] = new char*[BATCH_SIZE];
        data_temp[i] = new char*[BATCH_SIZE];
        for (int j = 0; j < BATCH_SIZE; ++j) {
            data[i][j] = nullptr;  // Initialize to null
            data_temp[i][j] = nullptr;
        }
    }

    // Read rows
    int row = 0;
    int batch_idx = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        int col = 0;
        int idx = 0;

        while (std::getline(ss, value, ',')) {
            value.erase(value.find_last_not_of(" \r\n\t") + 1);
            if(target_indices[idx] == col)
            {
                if (batch_idx==0) {
                    data[idx][row] = new char[value.length() + 1];
                    std::strcpy(data[idx][row], value.c_str());
                }
                else {
                    data_temp[idx][row] = new char[value.length() + 1];
                    std::strcpy(data_temp[idx][row], value.c_str());
                }
                idx++;
            }

            if(idx==numColumns)
                break;

            col++;
        }
        row++;
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
        for (int col = 0; col < numColumns; ++col) {
            if (data_temp[col]) {
                for (int row = 0; row < BATCH_SIZE; ++row) {
                    delete[] data_temp[col][row];  // delete each cell
                }
                delete[] data_temp[col];  // delete column
            }
        }
        delete[] data_temp;  
        data_temp = nullptr;
    }

    file.close();

    return true;
}  

bool Table::makeBatchFile(const int& batch_idx, char*** const &data_temp, const int& size) {
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
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < numColumns; ++j) {
                file << data[j][i];
                if (j != numColumns-1) file << ",";
            }
            file << "\n";
        }
    }
    else {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < numColumns; ++j) {
                file << data_temp[j][i];
                if (j != numColumns-1) file << ",";
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
        std::cerr << "GET BATCH: Failed to open file "<< file_path << std::endl;
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
            data[col][row] = new char[value.length() + 1];
            std::strcpy(data[col][row], value.c_str());
            col++;
        }
        row++;
    }

    file.close();

    return true;
}

int Table::getNumColumns() {
    return numColumns;
}

long long Table::getNumRows() {
    return numRows;
}

char** Table::getColumnNames() {
    return columnNames;
}

char*** Table::getData() {
    return data;
}

void Table::printData() {
    if (!data || numColumns == 0 || numRows == 0) {
        std::cout << "No data to print.\n";
        return;
    }
    std::cout << "--------- Printing Data for Table: "<<table_name<<"---------"<<std::endl;
    // Print header
    for (int col = 0; col < numColumns; ++col) {
        std::cout << columnNames[col];
        if (col < numColumns - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    // Print rows
    for (int row = 0; row < HEAD; ++row) {
        for (int col = 0; col < numColumns; ++col) {
            if (data[col][row]) {
                std::cout << data[col][row];
            } else {
                std::cout << "NULL";
            }

            if (col < numColumns - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
}

bool Table::readCSVColumns_DUCK(const std::string& filepath, const std::vector<std::string>& projections) {
    // Step 1: Allocate memory for the data buffer
    if (numRows == 0) {
        numColumns = projections.size();
        columnNames = new char*[numColumns];
        
        for(int i=0; i<numColumns; i++) {
            columnNames[i] = new char[projections[i].length() + 1];
            std::strcpy(columnNames[i], projections[i].c_str());
        }

        data = new char**[numColumns];
        for (int i = 0; i < numColumns; ++i) {
            data[i] = new char*[BATCH_SIZE];
            for (int j = 0; j < BATCH_SIZE; ++j) {
                data[i][j] = nullptr;
            }
        }

    } else {
        for (int i = 0; i < numColumns; ++i) {
            for (int j = 0; j < BATCH_SIZE; ++j) {
                data[i][j] = nullptr;
            }
        }
    }

    // Step 2: Set up in-memory DuckDB and build query
    duckdb::DuckDB db(nullptr); // in-memory DB
    duckdb::Connection con(db);

    std::string select_clause = "SELECT ";
    for (size_t i = 0; i < numColumns; ++i) {
        select_clause += "\"" + projections[i] + "\""; // handle spaces/special chars
        if (i < numColumns - 1) {
            select_clause += ", ";
        }
    }
    std::string query = select_clause + " FROM '" + filepath + "'";

    auto result = con.Query(query);
    //std::cout<<query<<std::endl;
    if (!result || result->HasError()) {
        std::cerr << "Query error: " << result->GetError() << std::endl;
        return false;
    }

    // Step 3: Read and store results
    size_t row_index = 0;
    while (auto chunk = result->Fetch()) {
        for (size_t i = 0; i < chunk->size(); i++) {
            for (size_t col = 0; col < numColumns; ++col) {
                std::string val = chunk->GetValue(col, i).ToString();
                data[col][row_index] = new char[val.length() + 1];
                std::strcpy(data[col][row_index], val.c_str());
            }
            row_index++;
            if (row_index >= BATCH_SIZE) break;
        }
        if (row_index >= BATCH_SIZE) break;
    }

    numRows += row_index;

    return true;
}


Table::~Table() {
    // Delete data
    if (data) {
        for (int col = 0; col < numColumns; ++col) {
            if (data[col]) {
                for (int row = 0; row < BATCH_SIZE; ++row) {
                    delete[] data[col][row];  // delete each cell
                }
                delete[] data[col];  // delete column
            }
        }
        delete[] data;  
        data = nullptr;
    }

    // Delete column names
    if (columnNames) {
        for (int col = 0; col < numColumns; ++col) {
            delete[] columnNames[col];
        }
        delete[] columnNames;
        columnNames = nullptr;
    }

    numColumns = 0;
    numRows = 0;
}
