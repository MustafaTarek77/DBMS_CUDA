#pragma once

#include <string>
#include <vector>
#include "config.hpp"
#include "utils.hpp"

class Table {
private:
    char** columnNames = nullptr;
    void** data;
    int numColumns = 0;
    long long numRows = 0;
    std::vector<int> target_indices;
    std::string table_name;

    bool makeTableBatches(const std::vector<std::string>& projections, const std::vector<std::string>& target_columns, const std::vector<Condition>& conditions); // Distibute the whole table into batches and fill data with the first batch
    bool makeBatches(const std::string& filepath, const std::vector<std::string>& projections, const std::vector<std::string>& target_columns, const std::vector<Condition>& conditions);
    bool makeBatchFile(const int& batch_idx, void** const &data_temp, const int& size);
    //bool readCSVColumns_DUCK(const std::string& filepath, const std::vector<std::string>& projections);
public:
    Table(const std::string &table_name, const std::vector<std::string>& projections, const std::vector<std::string>& target_columns, const std::vector<Condition>& conditions);
    bool getTableBatch(const int& batch_idx);
    int getNumColumns();
    long long getNumRows();
    void** getData();
    char** getColumnNames();
    void printData();
    ~Table();
};