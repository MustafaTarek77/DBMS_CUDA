#pragma once

#include "duckdb.hpp"
#include <string>
#include <vector>
#include <filesystem>
#include <unordered_map>
#include "duckdb/execution/physical_operator.hpp"
#include "Table.hpp"
#include "utils.hpp"
using namespace duckdb;

namespace fs = std::filesystem;

struct ColumnInfo {
    std::string name;
    std::string type;
    bool is_primary_key;
    bool is_foreign_key;
};

class DuckDBManager {
private:
    std::string csv_directory;
    std::string output_file;
    std::vector<fs::directory_entry> entries;
    std::unique_ptr<duckdb::DuckDB> db;
    std::unique_ptr<duckdb::Connection> con;
    std::vector<std::string> table_names;
    std::vector<PhysicalOperator*> execution_plan;
    bool is_aggregate = false;
    bool IS_GPU = true;

    Table* last_table_scanned_1 = nullptr;
    Table* last_table_scanned_2 = nullptr;
    bool turn = true;

    std::vector<ColumnInfo> GetCSVHeaders(const std::string &csv_file, const std::string &table_name, const std::vector<std::string> &table_names);
    std::string ConstructCreateTableQuery(const std::vector<ColumnInfo> &columns, const std::string &table_name);
    void TopologicalSortTables(const std::unordered_map<std::string, std::vector<std::string>> &dependencies);
    void TraversePlan(duckdb::PhysicalOperator *op);
    void ExecutePlan();
    void deleteLastTableScanned();
public:
    DuckDBManager(const std::string &csv_directory, const std::string &output_file, bool IS_GPU);
    void InitializeDatabase();
    void LoadTablesFromCSV();
    void AnalyzeQuery(const std::string &query);
    ~DuckDBManager();
};