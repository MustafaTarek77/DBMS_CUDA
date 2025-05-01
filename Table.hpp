#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#define FOLDER_PATH "./Data"
#define TABLE_PATH "./Tables/"
#define BATCH_SIZE 10000
#define HEAD 5

class Table {
private:
    char** columnNames = nullptr;
    char*** data;
    int numColumns = 0;
    long long numRows = 0;
    std::vector<int> target_indices;
    std::string table_name;

    bool makeTableBatches(const std::vector<std::string>& projections); // Distibute the whole table into batches and fill data with the first batch
    bool makeBatchFile(const int& batch_idx, char*** const &data_temp, const int& size);
    bool makeBatches(const std::string& filepath, const std::vector<std::string>& projections);
    bool readCSVColumns_DUCK(const std::string& filepath, const std::vector<std::string>& projections);
public:
    Table(const std::string &table_name, const std::vector<std::string>& projections);
    bool getTableBatch(const int& batch_idx);
    int getNumColumns();
    long long getNumRows();
    char*** getData();
    char** getColumnNames();
    void printData();
    ~Table();
};