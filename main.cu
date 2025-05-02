#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include "./DB.hpp"
#include "Table.hpp"
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    // DuckDBManager db_manager("./Data");

    // db_manager.InitializeDatabase();
    // db_manager.LoadTablesFromCSV();

    // std::string query = "SELECT s.name, s.age, a.address FROM Students s, Addresses a WHERE s.age>20 and s.year>1+1";
    // db_manager.AnalyzeQuery(query);

    std::string table_name = "test";
   
    std::vector<std::string> projections = {"ratio", "score", "name", "date_created"}; 
    std::vector<std::string> conditions = {};

    Table t(table_name, projections, conditions);
    t.printData();

    if (t.getTableBatch(7))
        t.printData();

    return 0;
}