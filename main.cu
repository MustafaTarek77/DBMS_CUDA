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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>\n";
        return 1;
    }

    std::string filename = argv[1];
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();  // Read entire file contents into buffer
    std::string query = buffer.str();

    std::cout << "Query read from file:\n" << query << "\n";

    DuckDBManager db_manager("./Data");

    db_manager.InitializeDatabase();
    db_manager.LoadTablesFromCSV();

    db_manager.AnalyzeQuery(query);

    // std::string table_name = "test";
   
    // std::vector<std::string> projections = {"ratio", "score", "name", "date_created"}; 
    // std::vector<std::string> conditions = {};

    // Table t(table_name, projections, conditions);
    // t.printData();

    // if (t.getTableBatch(7))
    //     t.printData();

    return 0;
}