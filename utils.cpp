#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem> 
namespace fs = std::filesystem;

#define FOLDER_PATH "./Data"

std::vector<std::pair<std::string, std::vector<std::string>>> readCSVColumns(const std::string& filepath) {
    std::ifstream file(filepath);
    std::vector<std::pair<std::string, std::vector<std::string>>> columns;

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return columns;
    }

    std::string line;
    
    // Read header line
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string header;
        
        // Parse headers
        while (std::getline(ss, header, ',')) {
            columns.push_back({header, {}});
        }
    }

    // Read the rest of the file
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        int col_idx = 0;
        
        while (std::getline(ss, value, ',')) {
            if (col_idx < columns.size()) {
                columns[col_idx].second.push_back(value);
            }
            col_idx++;
        }
    }

    file.close();
    return columns;
}

void scanTable(std::vector<std::pair<std::string, std::vector<std::string>>> &table, const std::string &table_name) {
    std::vector<std::string> filenames;

    for (const auto& entry : fs::directory_iterator(FOLDER_PATH)) {
        if (entry.path().extension() == ".csv") {
            filenames.push_back(entry.path().string());
        }
    }

    for (const auto& file_path : filenames) {
        // Get file name without extension
        std::string filename = fs::path(file_path).stem().string();

        if(filename==table_name)
        {
            table = readCSVColumns(file_path);
            return;
        }
    }
}