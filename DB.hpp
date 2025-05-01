#pragma once

#include "duckdb.hpp"
#include <string>
#include <vector>
#include <memory>
#include "Table.hpp"

struct ColumnInfo {
    std::string name;
    std::string type;
    bool is_primary_key;
    bool is_foreign_key;
};

struct LastTableScanned {
    char*** data;
    std::vector<std::string> columns_projections;
    std::string table_name;
};

class DuckDBManager {
private:
    std::string csv_directory;
    std::unique_ptr<duckdb::DuckDB> db;
    std::unique_ptr<duckdb::Connection> con;
    std::vector<std::string> table_names;

    LastTableScanned last_table_scanned_h;
    char*** last_op_input_d;
    char*** last_op_output_d;
    char*** output_h;

    std::vector<ColumnInfo> GetCSVHeaders(const std::string &csv_file, const std::vector<std::string> &table_names);
    std::string ConstructCreateTableQuery(const std::vector<ColumnInfo> &columns, const std::string &table_name);
    void TraversePlan(duckdb::PhysicalOperator *op);
    
public:
    DuckDBManager(const std::string &csv_directory);
    void InitializeDatabase();
    void LoadTablesFromCSV();
    void AnalyzeQuery(const std::string &query);
};