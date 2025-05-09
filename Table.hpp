#pragma once

#include <string>
#include <vector>
#include "config.hpp"
#include "utils.hpp"

class Table {
private:
    std::string FOLDER_PATH;
    std::string table_name;
    char** columnNames = nullptr; // target projections
    char** headers = nullptr; // all projections
    void** data;
    int numHeaders = 0;
    int numColumns = 0;
    int numRows = 0;
    int numBatches = 0;
    std::vector<int> projection_indices;
    std::vector<int> target_indices;

    bool makeTableBatches(const std::vector<std::string>& projections, const std::vector<std::string>& target_columns, const std::vector<std::vector<Condition>>& conditions); // Distibute the whole table into batches and fill data with the first batch
    bool makeBatches(const std::string& filepath, const std::vector<std::string>& projections, const std::vector<std::string>& target_columns, const std::vector<std::vector<Condition>>& conditions);
    bool makeBatchFile(void** const &data_temp, const int& size);
    bool checkConditions(const std::vector<std::vector<Condition>>& conditions, const std::string& row);
    bool compareFloats(float lhs, const std::string& op, float rhs);
    bool compareStrings(const std::string& lhs, const std::string& op, const std::string& rhs);
    bool compareDateTime(const std::string& lhs, const std::string& op, const std::string& rhs);
    //bool readCSVColumns_DUCK(const std::string& filepath, const std::vector<std::string>& projections);
public:
    Table(std::string FOLDER_PATH, const std::string &table_name, const std::vector<std::string>& projections, const std::vector<std::string>& target_columns, const std::vector<std::vector<Condition>>& conditions);
    Table(Table* table1, Table* table2, int* table1_indices, int* table2_indices, int total_rows, std::vector<std::string>& projections);
    Table(Table* table);
    bool getTableBatch(const int& batch_idx);
    std::string getTableName();
    int getNumColumns();
    int getNumRows();
    int getNumBatches();
    void** getData();
    char** getColumnNames();
    void printData();
    void writeDataToFile(std::string filename);
    ~Table();
};