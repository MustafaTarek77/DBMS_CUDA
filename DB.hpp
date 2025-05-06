#pragma once

#include "duckdb.hpp"
#include <string>
#include <vector>
#include "duckdb/execution/physical_operator.hpp"
#include "Table.hpp"
#include "utils.hpp"
using namespace duckdb;

struct ColumnInfo {
    std::string name;
    std::string type;
    bool is_primary_key;
    bool is_foreign_key;
};

class DuckDBManager {
private:
    std::string csv_directory;
    std::unique_ptr<duckdb::DuckDB> db;
    std::unique_ptr<duckdb::Connection> con;
    std::vector<std::string> table_names;
    std::vector<PhysicalOperator*> execution_plan;

    Table* last_table_scanned_h = nullptr;
    void** last_op_input_d;
    void** last_op_output_d;
    void** output_h;

    std::vector<ColumnInfo> GetCSVHeaders(const std::string &csv_file, const std::vector<std::string> &table_names);
    std::string ConstructCreateTableQuery(const std::vector<ColumnInfo> &columns, const std::string &table_name);
    void TraversePlan(duckdb::PhysicalOperator *op);
    void ExecutePlan();
    void deleteLastTableScanned();
public:
    DuckDBManager(const std::string &csv_directory);
    void InitializeDatabase();
    void LoadTablesFromCSV();
    void AnalyzeQuery(const std::string &query);
    ~DuckDBManager();
};