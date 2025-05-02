#pragma once

#include <string>
#include <vector>
#include <memory>
#include "config.hpp"

class Table {
private:
    char** columnNames = nullptr;
    void** data;
    int numColumns = 0;
    long long numRows = 0;
    std::vector<int> target_indices;
    std::string table_name;

    bool makeTableBatches(const std::vector<std::string>& projections, const std::vector<std::string>& conditions); // Distibute the whole table into batches and fill data with the first batch
    bool makeBatchFile(const int& batch_idx, void** const &data_temp, const int& size);
    bool makeBatches(const std::string& filepath, const std::vector<std::string>& projections, const std::vector<std::string>& conditions);
    //bool readCSVColumns_DUCK(const std::string& filepath, const std::vector<std::string>& projections);
public:
    Table(const std::string &table_name, const std::vector<std::string>& projections, const std::vector<std::string>& conditions);
    bool getTableBatch(const int& batch_idx);
    int getNumColumns();
    long long getNumRows();
    void** getData();
    char** getColumnNames();
    void printData();
    ~Table();
};