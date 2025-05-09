#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "./DB.hpp"
#include "Table.hpp"
#include <filesystem>
#include "config.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <filename>\n";
        return 1;
    }

    std::string FOLDER_PATH = "." + std::string(argv[1]);
    namespace fs = std::filesystem;
    if (!fs::exists(FOLDER_PATH) || !fs::is_directory(FOLDER_PATH)) {
        std::cerr << "Invalid folder path: " << FOLDER_PATH << "\n";
        return 1;
    }

    std::string FILENAME = argv[2];
    std::ifstream file(FILENAME);
    if (!file) {
        std::cerr << "Error: Could not open file " << FILENAME << "\n";
        return 1;
    }
 
    std::stringstream buffer;
    buffer << file.rdbuf();  // Read entire file contents into buffer
    std::string query = buffer.str();

    //std::cout << "Query read from file:\n" << query << "\n";

    fs::remove_all(TABLE_PATH);
    fs::create_directory(TABLE_PATH);

    std::string OUTPUT_FILE = "Team17_"+FILENAME.substr(0, FILENAME.find("."))+".csv";
    //std::cout<<FOLDER_PATH<<" "<<FILENAME<<" "<<OUTPUT_FILE<<std::endl;

    bool IS_GPU = true;
    DuckDBManager db_manager(FOLDER_PATH, OUTPUT_FILE, IS_GPU);
    db_manager.InitializeDatabase();
    db_manager.LoadTablesFromCSV();
    db_manager.AnalyzeQuery(query);

    // Calculate and print the execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Total execution time: " << duration << " milliseconds" << std::endl;
    std::cout << "                    = " << duration / 1000.0 << " seconds" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}