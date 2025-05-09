#include "./DB.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/execution/operator/order/physical_order.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/planner/table_binding.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/planner/bind_context.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/function/table_function.hpp" 
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include "db_utils.hpp"
#include "db_utils_cpu.hpp"

namespace fs = std::filesystem;
using namespace duckdb;


DuckDBManager::DuckDBManager(const std::string &csv_directory, const std::string &output_file, bool IS_GPU)
    : csv_directory(csv_directory), output_file(output_file), IS_GPU(IS_GPU), db(std::make_unique<DuckDB>(nullptr)), con(std::make_unique<Connection>(*db)) {}

void DuckDBManager::InitializeDatabase() {
    con->Query("SET disabled_optimizers = 'filter_pushdown, statistics_propagation';");
    // Collect all table names
    std::unordered_map<std::string, std::vector<std::string>> dependencies;

    for (const auto &entry : fs::directory_iterator(csv_directory)) {
        if (entry.path().extension() == ".csv") {
            std::string table_name = entry.path().stem().string();
            entries.push_back(entry);
            table_names.push_back(table_name);
        
            auto columns = GetCSVHeaders(entry.path().string(), table_name, table_names);
            for (const auto &col : columns) {
                if (col.is_foreign_key) {
                    std::string ref_table = col.name.substr(0, col.name.find('_'));
                    dependencies[table_name].push_back(ref_table);
                }
            }
        }
    }
    
    // Sort entries in topological order
    TopologicalSortTables(dependencies);
}

void DuckDBManager::TopologicalSortTables(const std::unordered_map<std::string, std::vector<std::string>> &dependencies)
{
    std::unordered_map<std::string, int> in_degree;
    for (const auto &table : table_names) {
        in_degree[table] = 0;
    }

    for (const auto &pair : dependencies) {
        for (const auto &dep : pair.second) {
            in_degree[pair.first]++;
        }
    }

    std::queue<std::string> q;
    for (const auto &pair : in_degree) {
        if (pair.second == 0) {
            q.push(pair.first);
        }
    }

    std::vector<std::string> sorted_tables;
    while (!q.empty()) {
        std::string current = q.front();
        q.pop();
        sorted_tables.push_back(current);

        for (const auto &pair : dependencies) {
            if (std::find(pair.second.begin(), pair.second.end(), current) != pair.second.end()) {
                in_degree[pair.first]--;
                if (in_degree[pair.first] == 0) {
                    q.push(pair.first);
                }
            }
        }
    }

    // Update the table order
    std::vector<fs::directory_entry> sorted_entries;
    for (const auto &name : sorted_tables) {
        for (const auto &entry : entries) {
            if (entry.path().stem().string() == name) {
                sorted_entries.push_back(entry);
                break;
            }
        }
    }

    entries = std::move(sorted_entries);
}


void DuckDBManager::LoadTablesFromCSV() {
    for (const auto &entry : entries) {
        if (entry.path().extension() == ".csv") {
            std::string csv_file = entry.path().string();
            std::string table_name = entry.path().stem().string();

            auto columns = GetCSVHeaders(csv_file, table_name, table_names);
            std::string create_query = ConstructCreateTableQuery(columns, table_name);

            //std::cout << "Executing: " << create_query << std::endl;
            auto result = con->Query(create_query);
            if (result->HasError()) {
                std::cerr << "Failed to execute query: " << result->GetError() << std::endl;
            } else {
                //std::cout << "Query executed successfully for table: " << table_name << std::endl;
            }          
        }
    }
}

std::vector<ColumnInfo> DuckDBManager::GetCSVHeaders(const std::string &csv_file, const std::string &table_name, const std::vector<std::string> &table_names) {
    std::ifstream file(csv_file);
    std::string line;
    std::vector<ColumnInfo> columns;

    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string header;
        while (std::getline(ss, header, ',')) {
            ColumnInfo info;
            info.is_primary_key = false;
            info.is_foreign_key = false;

            if (header.find("(P)") != std::string::npos) {
                info.name = header.substr(0, header.find(" (P)"));
                info.is_primary_key = true;
            }

            if (header.find("(N)") != std::string::npos) {
                info.type = "FLOAT"; 
                info.name = header.substr(0, header.find(" (N)"));
            } else if (header.find("(T)") != std::string::npos) {
                info.type = "VARCHAR";
                info.name = header.substr(0, header.find(" (T)"));
            } else if (header.find("(D)") != std::string::npos) {
                info.type = "TIMESTAMP";
                info.name = header.substr(0, header.find(" (D)"));
            } else {
                info.type = "VARCHAR";
                info.name = header;
            }

            for (const auto &table : table_names) {
                if (header.find(table + "_") != std::string::npos && table!=table_name) {
                    info.is_foreign_key = true;
                    break;
                }
            }

            info.name.erase(info.name.find_last_not_of(" \n\r\t") + 1);

            columns.push_back(info);
        }
    }

    return columns;
}

std::string DuckDBManager::ConstructCreateTableQuery(const std::vector<ColumnInfo> &columns, const std::string &table_name) {
    std::string query = "CREATE TABLE " + table_name + " (";
    std::string primary_key_column;
    std::vector<std::string> foreign_keys;

    for (size_t i = 0; i < columns.size(); ++i) {
        query += columns[i].name + " " + columns[i].type;
        if (i != columns.size() - 1) {
            query += ", ";
        }
        if (columns[i].is_primary_key) {
            primary_key_column = columns[i].name;
        }
        if (columns[i].is_foreign_key) {
            foreign_keys.push_back(columns[i].name);
        }
    }

    if (!primary_key_column.empty()) {
        query += ", PRIMARY KEY (" + primary_key_column + ")";
    }

    for (const auto &key : foreign_keys) {
        size_t pos = key.find('_');
        if (pos != std::string::npos) {
            std::string referenced_table = key.substr(0, pos);
            //std::string referenced_column = key.substr(pos + 1);

            query += ", FOREIGN KEY (" + key + ") REFERENCES " + referenced_table + " (" + key + ")";
        }
    }

    query += ");";
    return query;
}

void DuckDBManager::AnalyzeQuery(const std::string &query) {
    Parser parser;
    parser.ParseQuery(query);
    if (parser.statements.empty()) {
        std::cerr << "Parsing failed!" << std::endl;
        return;
    }

    auto &ctx = *con->context;
    ctx.transaction.BeginTransaction();
    // Step 2: Create logical plan
    Planner planner(ctx);
    planner.CreatePlan(std::move(parser.statements[0]));

    // Step 3: Optimize logical plan
    Optimizer optimizer(*planner.binder, ctx);
    auto logical_plan = optimizer.Optimize(std::move(planner.plan));

    // Step 4: Create physical plan
    PhysicalPlanGenerator physical_generator(ctx);
    
    auto physical_plan = physical_generator.Plan(logical_plan->Copy(ctx));
    
    //std::cout << "=== PHYSICAL PLAN ===" << std::endl;
    
    //std::cout << physical_plan.get()->Root().ToString() << std::endl;   
    TraversePlan(&physical_plan.get()->Root());
    ExecutePlan();
    if(!is_aggregate) {
        last_table_scanned_1->writeDataToFile(output_file);
    }
}

void DuckDBManager::TraversePlan(PhysicalOperator *op) {
    for (auto &child : op->children) {
        TraversePlan(&child.get());
    }
    execution_plan.push_back(op);
}

void DuckDBManager::ExecutePlan() {
    for(int i=0; i<execution_plan.size(); i++) {
        switch(execution_plan[i]->type) {
            case PhysicalOperatorType::TABLE_SCAN: {
                auto params = execution_plan[i]->ParamsToString();
                std::string table_name;
                std::vector<std::string> projections, target_columns;
                std::vector<std::vector<Condition>> conditions;

                // Getting table name and initial projections
                for(auto &param: params)
                {
                    if(param.first=="Table") {
                        //std::cout<<"Scanning Table:- "<<param.second<<std::endl;
                        table_name = param.second;
                    }
                    else if(param.first=="Projections") {
                        //std::cout << "Projections: ";
                        std::istringstream iss(param.second);
                        std::string projection;
                        while (std::getline(iss, projection)) {
                            if (!projection.empty()) {
                                //std::cout << projection << " ";
                                projections.push_back(projection);
                            }
                        }
                        //std::cout << std::endl;
                    }
                }
                target_columns = projections;
                
                // Getting Conditions if exist
                if(i+1<execution_plan.size() && execution_plan[i+1]->type==PhysicalOperatorType::FILTER) {
                    auto params = execution_plan[i+1]->ParamsToString();
                    //std::cout << "Filters:-" << std::endl;
                
                    for (auto &param : params) {
                        if (param.first=="__expression__") {
                            std::string filter_str = param.second;
                            //std::cout << "Processing filter string: "<< filter_str << std::endl;
                            parseComplexExpression(filter_str, conditions);
                            // printConditionStructure(conditions);
                        }
                    }
                    i++;
                }

                // Getting Projections if exist
                if(i+1<execution_plan.size() && execution_plan[i+1]->type==PhysicalOperatorType::PROJECTION) {
                    std::vector<std::string> columns;
                    auto params = execution_plan[i+1]->ParamsToString();
                    //std::cout << "Projecting expressions:- ";
                    for(auto &param: params) {
                        if (param.first=="__projections__" && param.second.substr(0, param.second.find("("))!="CAST") {
                            std::istringstream iss(param.second);
                            std::string projection;
                            while (std::getline(iss, projection)) {
                                if (!projection.empty()) {
                                    if(projection[0]=='#') {
                                        std::string proj = projection.substr(1);
                                        //std::cout << projections[std::stoi(proj)] << " ";
                                        columns.push_back(projections[std::stoi(proj)]);
                                    }
                                    else {
                                        //std::cout << projection << " ";
                                        columns.push_back(projection);
                                    }
                                }
                            }
                        }
                    }
                    //std::cout << std::endl;
                    if (columns.size()!=0) {
                        target_columns.clear();
                        target_columns = columns;
                    }
                    i++;
                }

                if (turn == true) {
                    delete last_table_scanned_1;
                    last_table_scanned_1 = nullptr; 
                    last_table_scanned_1 = new Table(csv_directory, table_name, projections, target_columns, conditions);
                    //last_table_scanned_1->printData();
                    //std::cout<<"Table is scanned successfully"<<std::endl;
                    turn = false;
                }
                else {
                    delete last_table_scanned_2;
                    last_table_scanned_2 = nullptr; 
                    last_table_scanned_2 = new Table(csv_directory, table_name, projections, target_columns, conditions);
                    //last_table_scanned_2->printData();
                    //std::cout<<"Table is scanned successfully"<<std::endl;
                    turn = true;
                }
                break;
            }
            case PhysicalOperatorType::HASH_JOIN: {
                Table* new_table1 = new Table(last_table_scanned_1);
                Table* new_table2 = new Table(last_table_scanned_2);
                deleteLastTableScanned();
                last_table_scanned_1 = new_table1;
                last_table_scanned_2 = new_table2;
                std::vector<std::string> target_columns;                
                std::vector<std::string> projections;
                char** names1 = last_table_scanned_1->getColumnNames();
                for (int j=0; j<last_table_scanned_1->getNumColumns(); j++) {
                    std::string header = std::string(names1[j]);
                    std::string column = header.substr(0, header.find(" ("));
                    projections.push_back(column);
                }
                char** names2 = last_table_scanned_2->getColumnNames();
                for (int j=0; j<last_table_scanned_2->getNumColumns(); j++) {
                    std::string header = std::string(names2[j]);
                    std::string column = header.substr(0, header.find(" ("));
                    projections.push_back(column);
                }
                target_columns = projections;

                auto &join = execution_plan[i]->Cast<PhysicalHashJoin>();
                size_t num_conditions = join.conditions.size();
                Kernel_Condition** conditions = new Kernel_Condition*[1];
                conditions[0] = new Kernel_Condition[num_conditions];
                for (int c=0; c<num_conditions; c++) {
                    std::string left_operand = join.conditions[c].left->ToString();
                    std::string right_operand = join.conditions[c].right->ToString();
                    for (int j=0; j<last_table_scanned_1->getNumColumns(); j++) {
                        std::string header = std::string(names1[j]);
                        std::string column = header.substr(0, header.find(" ("));
                        if(column==left_operand) {
                            conditions[0][c].idx1 = j;
                            target_columns[j]=" ";
                            break;
                        }
                    }
                    bool flag = false;
                    for (int j=0; j<last_table_scanned_2->getNumColumns(); j++) {
                        std::string header = std::string(names2[j]);
                        std::string column = header.substr(0, header.find(" ("));
                        if(column==right_operand) {
                            conditions[0][c].idx2 = j;
                            target_columns[j]=" ";
                            flag = true;
                            break;
                        }
                    }
                    duckdb::ExpressionType expr_type = join.conditions[c].comparison;  
                    switch (expr_type) {
                        case duckdb::ExpressionType::COMPARE_EQUAL: conditions[0][c].relational_operator = '='; break;
                        case duckdb::ExpressionType::COMPARE_LESSTHAN: conditions[0][c].relational_operator = '<'; break;
                        case duckdb::ExpressionType::COMPARE_GREATERTHAN: conditions[0][c].relational_operator = '>'; break;
                        case duckdb::ExpressionType::COMPARE_NOTEQUAL: conditions[0][c].relational_operator = '!'; break;
                        default: 
                        std::cout << "ERROR: Unhandeled Operation"<<std::endl;
                        conditions[0][c].relational_operator = '?';
                    }

                    char** names = last_table_scanned_1->getColumnNames();
                    for (int j=0; j<last_table_scanned_1->getNumColumns(); j++) {
                        std::string header = std::string(names[j]);
                        if (header.substr(0, header.find(" ("))==left_operand) {
                            if (header.find("(N)") != std::string::npos) {
                                conditions[0][c].type = 'N';
                                if (flag==false) {
                                    conditions[0][c].idx2 = -1;
                                    conditions[0][c].value = new float(std::stof(right_operand));
                                }
                            } 
                            else if (header.find("(T)") != std::string::npos) {
                                conditions[0][c].type = 'T';
                                if (flag==false) {
                                    conditions[0][c].idx2 = -1;
                                    conditions[0][c].value = new char[right_operand.length() + 1];
                                    std::strcpy(static_cast<char*>(conditions[0][c].value), right_operand.c_str());
                                }
                            }
                            else if (header.find("(D)") != std::string::npos) {
                                conditions[0][c].type = 'D';
                                if (flag==false) {
                                    conditions[0][c].idx2 = -1;
                                    conditions[0][c].value = new char[right_operand.length() + 1];
                                    std::strcpy(static_cast<char*>(conditions[0][c].value), right_operand.c_str());
                                }
                            }
                        }
                    }

                    //std::cout << conditions[0][c].type << " Join on condition: " << conditions[0][c].idx1
                    //         << conditions[0][c].relational_operator << conditions[0][c].idx2 << " --> "<< conditions[0][c].value << std::endl;
                }

                // Getting Projections if exist
                if(i+1<execution_plan.size() && execution_plan[i+1]->type==PhysicalOperatorType::PROJECTION) {
                    std::vector<std::string> columns;
                    auto params = execution_plan[i+1]->ParamsToString();
                    //std::cout << "Projecting expressions:- ";
                    for(auto &param: params) {
                        if (param.first=="__projections__" && param.second.substr(0, param.second.find("("))!="CAST") {
                            std::istringstream iss(param.second);
                            std::string projection;
                            while (std::getline(iss, projection)) {
                                if (!projection.empty()) {
                                    if(projection[0]=='#') {
                                        std::string proj = projection.substr(1);
                                        int idx = std::stoi(proj)+1;
                                        if(idx<projections.size()) {
                                            //std::cout << projections[idx] << " ";
                                            columns.push_back(projections[idx]);
                                        }
                                        else {
                                            //std::cout << projections[idx-1] << " ";
                                            columns.push_back(projections[idx-1]);
                                        }
                                    }
                                    else {
                                        //std::cout << projection << " ";
                                        columns.push_back(projection);
                                    }
                                }
                            }
                        }
                    }
                    //std::cout << std::endl;
                    if (columns.size()!=0) {
                        target_columns = columns;
                    }
                    else {
                        target_columns = projections;
                    }
                    i++;
                }
                std::vector<int> conditions_groups_sizes;
                conditions_groups_sizes.push_back(num_conditions);
                int *out1, *out2;
                int result_count;
                if (IS_GPU) {
                    ExecuteJoin(conditions, num_conditions, conditions_groups_sizes, target_columns, last_table_scanned_1, last_table_scanned_2, out1, out2, result_count);
                }
                else {
                    ExecuteJoinCPU(conditions, num_conditions, conditions_groups_sizes, target_columns, last_table_scanned_1, last_table_scanned_2, out1, out2, result_count);
                }
                // std::cout << "Matched Row Indices:\n";
                // for (int k = 0; k < result_count; ++k) {
                //     std::cout << "Pair " << k << ": out1[" << k << "] = " << out1[k]
                //               << ", out2[" << k << "] = " << out2[k] << '\n';
                // }
                Table* new_table = new Table(last_table_scanned_1, last_table_scanned_2, out1, out2, result_count, target_columns);
                deleteLastTableScanned();
                last_table_scanned_1 = new_table;
                turn = false;
                //last_table_scanned_1->printData();

                delete[] out1;
                delete[] out2;
                for (int g = 0; g < conditions_groups_sizes.size(); ++g) {
                    for (int k = 0; k < conditions_groups_sizes[g]; ++k) {
                        Kernel_Condition& cond = conditions[g][k];
                
                        if (cond.value != nullptr) {
                            if (cond.type == 'N') {
                                delete static_cast<float*>(cond.value);
                            }
                            else if (cond.type == 'T' || cond.type == 'D') {
                                delete[] static_cast<char*>(cond.value);  // assuming these are C-strings
                            }
                            cond.value = nullptr; // prevent dangling pointer
                        }
                    }
                }
                break;
            }
            case PhysicalOperatorType::NESTED_LOOP_JOIN: {
                Table* new_table1 = new Table(last_table_scanned_1);
                Table* new_table2 = new Table(last_table_scanned_2);
                deleteLastTableScanned();
                last_table_scanned_1 = new_table1;
                last_table_scanned_2 = new_table2;
                std::vector<std::string> target_columns;                
                std::vector<std::string> projections;
                char** names1 = last_table_scanned_1->getColumnNames();
                for (int j=0; j<last_table_scanned_1->getNumColumns(); j++) {
                    std::string header = std::string(names1[j]);
                    std::string column = header.substr(0, header.find(" ("));
                    projections.push_back(column);
                }
                char** names2 = last_table_scanned_2->getColumnNames();
                for (int j=0; j<last_table_scanned_2->getNumColumns(); j++) {
                    std::string header = std::string(names2[j]);
                    std::string column = header.substr(0, header.find(" ("));
                    projections.push_back(column);
                }
                target_columns = projections;

                auto &join = execution_plan[i]->Cast<PhysicalNestedLoopJoin>();
                size_t num_conditions = join.conditions.size();
                Kernel_Condition** conditions = new Kernel_Condition*[1];
                conditions[0] = new Kernel_Condition[num_conditions];
                for (int c=0; c<num_conditions; c++) {
                    std::string left_operand = join.conditions[c].left->ToString();
                    std::string right_operand = join.conditions[c].right->ToString();
                    for (int j=0; j<last_table_scanned_1->getNumColumns(); j++) {
                        std::string header = std::string(names1[j]);
                        std::string column = header.substr(0, header.find(" ("));
                        if(column==left_operand) {
                            conditions[0][c].idx1 = j;
                            break;
                        }
                    }
                    bool flag = false;
                    for (int j=0; j<last_table_scanned_2->getNumColumns(); j++) {
                        std::string header = std::string(names2[j]);
                        std::string column = header.substr(0, header.find(" ("));
                        if(column==right_operand) {
                            conditions[0][c].idx2 = j;
                            flag = true;
                            break;
                        }
                    }
                    duckdb::ExpressionType expr_type = join.conditions[c].comparison;  
                    switch (expr_type) {
                        case duckdb::ExpressionType::COMPARE_EQUAL: conditions[0][c].relational_operator = '='; break;
                        case duckdb::ExpressionType::COMPARE_LESSTHAN: conditions[0][c].relational_operator = '<'; break;
                        case duckdb::ExpressionType::COMPARE_GREATERTHAN: conditions[0][c].relational_operator = '>'; break;
                        case duckdb::ExpressionType::COMPARE_NOTEQUAL: conditions[0][c].relational_operator = '!'; break;
                        default: 
                        std::cout << "ERROR: Unhandeled Operation"<<std::endl;
                        conditions[0][c].relational_operator = '?';
                    }

                    for (int j=0; j<last_table_scanned_1->getNumColumns(); j++) {
                        std::string header = std::string(names1[j]);
                        if (header.substr(0, header.find(" ("))==left_operand) {
                            if (header.find("(N)") != std::string::npos) {
                                conditions[0][c].type = 'N';
                                if (flag==false) {
                                    conditions[0][c].idx2 = -1;
                                    conditions[0][c].value = new float(std::stof(right_operand));
                                }
                            } 
                            else if (header.find("(T)") != std::string::npos) {
                                conditions[0][c].type = 'T';
                                if (flag==false) {
                                    conditions[0][c].idx2 = -1;
                                    conditions[0][c].value = new char[right_operand.length() + 1];
                                    std::strcpy(static_cast<char*>(conditions[0][c].value), right_operand.c_str());
                                }
                            }
                            else if (header.find("(D)") != std::string::npos) {
                                conditions[0][c].type = 'D';
                                if (flag==false) {
                                    conditions[0][c].idx2 = -1;
                                    conditions[0][c].value = new char[right_operand.length() + 1];
                                    std::strcpy(static_cast<char*>(conditions[0][c].value), right_operand.c_str());
                                }
                            }
                        }
                    }

                    //std::cout << conditions[0][c].type << " Join on condition: " << conditions[0][c].idx1
                    //          << conditions[0][c].relational_operator << conditions[0][c].idx2 << " --> " << conditions[0][c].value << std::endl;
                }

                // Getting Projections if exist
                if(i+1<execution_plan.size() && execution_plan[i+1]->type==PhysicalOperatorType::PROJECTION) {
                    std::vector<std::string> columns;
                    auto params = execution_plan[i+1]->ParamsToString();
                    //std::cout << "Projecting expressions:- ";
                    for(auto &param: params) {
                        if (param.first=="__projections__" && param.second.substr(0, param.second.find("("))!="CAST") {
                            std::istringstream iss(param.second);
                            std::string projection;
                            while (std::getline(iss, projection)) {
                                if (!projection.empty()) {
                                    if(projection[0]=='#') {
                                        std::string proj = projection.substr(1);
                                        //std::cout << projections[std::stoi(proj)] << " ";
                                        columns.push_back(projections[std::stoi(proj)]);
                                    }
                                    else {
                                        //std::cout << projection << " ";
                                        columns.push_back(projection);
                                    }
                                }
                            }
                        }
                    }
                    //std::cout << std::endl;
                    if (columns.size()!=0) {
                        target_columns = columns;
                    }
                    else {
                        target_columns = projections;
                    }
                    i++;
                }
                std::vector<int> conditions_groups_sizes;
                conditions_groups_sizes.push_back(num_conditions);
                int *out1, *out2;
                int result_count;
                if (IS_GPU) {
                    ExecuteJoin(conditions, num_conditions, conditions_groups_sizes, target_columns, last_table_scanned_1, last_table_scanned_2, out1, out2, result_count);
                }
                else {
                    ExecuteJoinCPU(conditions, num_conditions, conditions_groups_sizes, target_columns, last_table_scanned_1, last_table_scanned_2, out1, out2, result_count);
                }       
                // std::cout << "Matched Row Indices:\n";
                // for (int k = 0; k < result_count; ++k) {
                //     std::cout << "Pair " << k << ": out1[" << k << "] = " << out1[k]
                //               << ", out2[" << k << "] = " << out2[k] << '\n';
                // }
                Table* new_table = new Table(last_table_scanned_1, last_table_scanned_2, out1, out2, result_count, target_columns);
                deleteLastTableScanned();
                last_table_scanned_1 = new_table;
                turn = false;
                //last_table_scanned_1->printData();

                delete[] out1;
                delete[] out2;
                for (int g = 0; g < conditions_groups_sizes.size(); ++g) {
                    for (int k = 0; k < conditions_groups_sizes[g]; ++k) {
                        Kernel_Condition& cond = conditions[g][k];
                
                        if (cond.value != nullptr) {
                            if (cond.type == 'N') {
                                delete static_cast<float*>(cond.value);
                            }
                            else if (cond.type == 'T' || cond.type == 'D') {
                                delete[] static_cast<char*>(cond.value);  // assuming these are C-strings
                            }
                            cond.value = nullptr; // prevent dangling pointer
                        }
                    }
                }
                break;
            }
            case PhysicalOperatorType::CROSS_PRODUCT: {
                Table* new_table1 = new Table(last_table_scanned_1);
                Table* new_table2 = new Table(last_table_scanned_2);
                deleteLastTableScanned();
                last_table_scanned_1 = new_table1;
                last_table_scanned_2 = new_table2;
                std::vector<std::string> target_columns;                
                std::vector<std::vector<Condition>> conditions;
                std::vector<std::string> projections;
                char** names1 = last_table_scanned_1->getColumnNames();
                for (int j=0; j<last_table_scanned_1->getNumColumns(); j++) {
                    std::string header = std::string(names1[j]);
                    std::string column = header.substr(0, header.find(" ("));
                    projections.push_back(column);
                }
                char** names2 = last_table_scanned_2->getColumnNames();
                for (int j=0; j<last_table_scanned_2->getNumColumns(); j++) {
                    std::string header = std::string(names2[j]);
                    std::string column = header.substr(0, header.find(" ("));
                    projections.push_back(column);
                }
                target_columns = projections;
                
                // Getting Conditions if exist
                if(i+1<execution_plan.size() && execution_plan[i+1]->type==PhysicalOperatorType::FILTER) {
                    auto params = execution_plan[i+1]->ParamsToString();
                    //std::cout << "Filters:-" << std::endl;
                
                    for (auto &param : params) {
                        if (param.first=="__expression__") {
                            std::string filter_str = param.second;
                        //    std::cout << "Processing filter string: "<< filter_str << std::endl;
                            parseComplexExpression(filter_str, conditions);
                        }
                    }
                    i++;
                }

                std::vector<int> conditions_groups_sizes;
                size_t num_group_conditions = conditions.size();
                int total_conditions = 0; 
                Kernel_Condition** kernel_conditions = new Kernel_Condition*[num_group_conditions];
                for (size_t c = 0; c < num_group_conditions; ++c) {
                    size_t inner_size = conditions[c].size();
                    kernel_conditions[c] = new Kernel_Condition[inner_size];
                    total_conditions += inner_size;
                    conditions_groups_sizes.push_back(inner_size);
                    for (size_t j = 0; j < inner_size; ++j) {
                        const Condition& cond = conditions[c][j];
                        Kernel_Condition& kcond = kernel_conditions[c][j];
                        bool flag = false;
                        for (int k=0; k<last_table_scanned_2->getNumColumns(); k++) {
                            std::string header = std::string(names2[k]);
                            std::string column = header.substr(0, header.find(" ("));
                            if(column==cond.right_operand) {
                                kcond.idx2 = k;
                                flag = true;
                                break;
                            }
                        }
                        for (int k=0; k<last_table_scanned_1->getNumColumns(); k++) {
                            std::string header = std::string(names1[k]);
                            std::string column = header.substr(0, header.find(" ("));
                            if(column==cond.left_operand) {
                                kcond.idx1 = k;
                                if (header.find("(N)") != std::string::npos) {
                                    kcond.type = 'N';
                                    if (flag==false) {
                                        kcond.idx2 = -1;
                                        kcond.value = new float(std::stof(cond.right_operand));
                                    }
                                } 
                                else if (header.find("(T)") != std::string::npos) {
                                    kcond.type = 'T';
                                    if (flag==false) {
                                        kcond.idx2 = -1;
                                        kcond.value = new char[cond.right_operand.length() + 1];
                                        std::strcpy(static_cast<char*>(kcond.value), cond.right_operand.c_str());
                                    }
                                }
                                else if (header.find("(D)") != std::string::npos) {
                                    kcond.type = 'D';
                                    if (flag==false) {
                                        kcond.idx2 = -1;
                                        kcond.value = new char[cond.right_operand.length() + 1];
                                        std::strcpy(static_cast<char*>(kcond.value), cond.right_operand.c_str());
                                    }
                                }
                                break;
                            }
                        }
                        
                        kcond.relational_operator = cond.relational_operator.empty() ? '?' : cond.relational_operator[0];
                        //std::cout << kcond.type << " Join on condition: " << kcond.idx1
                        //      << kcond.relational_operator << kcond.idx2 << " --> " << kcond.value << std::endl;
                    }
                }

                // Getting Projections if exist
                if(i+1<execution_plan.size() && execution_plan[i+1]->type==PhysicalOperatorType::PROJECTION) {
                    std::vector<std::string> columns;
                    auto params = execution_plan[i+1]->ParamsToString();
                    //std::cout << "Projecting expressions:- ";
                    for(auto &param: params) {
                        if (param.first=="__projections__" && param.second.substr(0, param.second.find("("))!="CAST") {
                            std::istringstream iss(param.second);
                            std::string projection;
                            while (std::getline(iss, projection)) {
                                if (!projection.empty()) {
                                    if(projection[0]=='#') {
                                        std::string proj = projection.substr(1);
                                    //    std::cout << projections[std::stoi(proj)] << " ";
                                        columns.push_back(projections[std::stoi(proj)]);
                                    }
                                    else {
                                    //    std::cout << projection << " ";
                                        columns.push_back(projection);
                                    }
                                }
                            }
                        }
                    }
                    //std::cout << std::endl;
                    if (columns.size()!=0) {
                        target_columns = columns;
                    }
                    else {
                        target_columns = projections;
                    }
                    i++;
                }
                int *out1, *out2;
                int result_count;
                if (IS_GPU) {
                    ExecuteJoin(kernel_conditions, total_conditions, conditions_groups_sizes, target_columns, last_table_scanned_1, last_table_scanned_2, out1, out2, result_count);
                }
                else {
                    ExecuteJoinCPU(kernel_conditions, total_conditions, conditions_groups_sizes, target_columns, last_table_scanned_1, last_table_scanned_2, out1, out2, result_count);
                }
                // std::cout << "Matched Row Indices:\n";
                // for (int k = 0; k < result_count; ++k) {
                //     std::cout << "Pair " << k << ": out1[" << k << "] = " << out1[k]
                //               << ", out2[" << k << "] = " << out2[k] << '\n';
                // }
                Table* new_table = new Table(last_table_scanned_1, last_table_scanned_2, out1, out2, result_count, target_columns);
                deleteLastTableScanned();
                last_table_scanned_1 = new_table;
                turn = false;
                //last_table_scanned_1->printData();

                delete[] out1;
                delete[] out2;                
                for (int g = 0; g < conditions_groups_sizes.size(); ++g) {
                    for (int k = 0; k < conditions_groups_sizes[g]; ++k) {     
                        Kernel_Condition& cond = kernel_conditions[g][k];
                        if (cond.value != nullptr || static_cast<void*>(cond.value)!=0) {
                            if (cond.type == 'N') {
                                delete static_cast<float*>(cond.value);
                            }
                            else if (cond.type == 'T' || cond.type == 'D') {
                                delete[] static_cast<char*>(cond.value);
                            }
                            cond.value = nullptr;
                        }
                    }
                }     
                break;
            }
            case PhysicalOperatorType::UNGROUPED_AGGREGATE: {
                auto params = execution_plan[i]->ParamsToString();
                //std::cout << "Applying Aggregation:-" << std::endl;
                
                std::vector<std::string> agg_functions;
                std::vector<std::string> agg_targets;
                
                for (auto &param : params) {
                    if (param.first == "Aggregates") {
                        //std::cout << "Aggregate Functions: " << std::endl;

                        std::istringstream iss(param.second);
                        std::string agg_expr;
                        while (std::getline(iss, agg_expr)) {
                            if (!agg_expr.empty()) {
                                size_t openParen = agg_expr.find('(');
                                size_t closeParen = agg_expr.find(')');

                                if (openParen == std::string::npos || closeParen == std::string::npos || agg_expr[openParen + 1] != '#') {
                                    std::cerr << "Invalid aggregate format: " << agg_expr << std::endl;
                                    continue;
                                }

                                std::string function = agg_expr.substr(0, openParen); 
                                std::string columnIndex = agg_expr.substr(openParen + 2, closeParen - openParen - 2); 
                                int columnIdx = std::stoi(columnIndex);

                                agg_functions.push_back(function);
                                agg_targets.push_back(columnIndex);

                                //std::cout << "  Function: " << function << ", Column Index: " << columnIdx << std::endl;
                                
                                
                                ExecuteAggregateFunction(function, columnIdx, last_table_scanned_1, output_file, IS_GPU);
                                is_aggregate = true;
                            }
                        }
                    }
                }
                break;
            }
            case PhysicalOperatorType::ORDER_BY: {
                // Table* new_table1 = new Table(last_table_scanned_1);
                // deleteLastTableScanned();
                // last_table_scanned_1 = new_table1;
                auto params = execution_plan[i]->ParamsToString();
                int orderByColumnIndex=-1;
                bool orderByAscending=1;
                bool flag = true;
                for (auto& param : params) {
                    //std::cout<<param.first<<", "<<param.second<<std::endl;
                    if (param.first == "__order_by__") {
                        // Parse column name and ordering from param.second
                        std::string paramString = param.second; 
                        std::string columnName;
                        bool isAscending = true; // Default to ascending

                        // Extract column name - always after the final dot
                        size_t lastDotPos = paramString.find_last_of('.');
                        if (lastDotPos != std::string::npos) {
                            // Check if the column name is quoted
                            if (lastDotPos + 1 < paramString.length() && paramString[lastDotPos + 1] == '\"') {
                                size_t quoteStart = lastDotPos + 1;
                                size_t quoteEnd = paramString.find('\"', quoteStart + 1);
                                
                                if (quoteEnd != std::string::npos) {
                                    columnName = paramString.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
                                }
                            } else {
                                size_t spacePos = paramString.find(' ', lastDotPos);
                                if (spacePos != std::string::npos) {
                                    columnName = paramString.substr(lastDotPos + 1, spacePos - lastDotPos - 1);
                                } else {
                                    columnName = paramString.substr(lastDotPos + 1);
                                }
                            }
                            flag = true;
                        } else {
                            // std::istringstream iss(param.second);
                            std::string projection = param.second;
                            // while (std::getline(iss, projection)) {
                                if (!projection.empty()) {
                                    if(projection[0]=='#') {
                                        std::string proj = projection.substr(1);
                                        //std::cout << last_table_scanned_1->getColumnNames()[std::stoi(proj)] << " \n";
                                        columnName = last_table_scanned_1->getColumnNames()[std::stoi(proj)];
                                    }
                                    else {
                                        //std::cout << projection << " \n";
                                        columnName = projection;
                                    }
                                    flag = false;
                                }
                            //}
                        }

                        // Determine if ASC or DESC
                        std::string upperParam = paramString;
                        std::transform(upperParam.begin(), upperParam.end(), upperParam.begin(), ::toupper);

                        if (upperParam.find(" DESC") != std::string::npos) {
                            isAscending = false;
                        }

                        //std::cout << "Column name to find: '" << columnName << "'" << std::endl;
                        //std::cout << "Order: " << (isAscending ? "ASC" : "DESC") << std::endl;

                        // Find the column index in the table
                        int columnIndex = -1;
                        if (last_table_scanned_1 != nullptr) {
                            char** columnNames = last_table_scanned_1->getColumnNames();
                            int numColumns = last_table_scanned_1->getNumColumns();

                            for (int j = 0; j < numColumns; j++) {
                                if (columnNames[j] != nullptr) {
                                    std::string fullColName = columnNames[j];
                                    if (flag==true) {
                                        std::string colNameOnly;
                                        
                                        size_t parenPos = fullColName.find(" (");
                                        if (parenPos != std::string::npos) {
                                            colNameOnly = fullColName.substr(0, parenPos);
                                        } else {
                                            colNameOnly = fullColName;
                                        }

                                        std::string colNameLower = colNameOnly;
                                        std::string targetNameLower = columnName;
                                        std::transform(colNameLower.begin(), colNameLower.end(), colNameLower.begin(), ::tolower);
                                        std::transform(targetNameLower.begin(), targetNameLower.end(), targetNameLower.begin(), ::tolower);

                                        if (colNameLower == targetNameLower) {
                                            columnIndex = j;
                                            break;
                                        }
                                    }
                                    else {
                                        if (fullColName == columnName) {
                                            columnIndex = j;
                                            break;
                                        }
                                    }
                                }
                            }
                            
                            // if (columnIndex != -1) {
                            //     std::cout << "Column index for sorting: " << columnIndex << std::endl;
                            // } else {
                            //     std::cout << "Column not found in table" << std::endl;
                            // }
                            
                            orderByColumnIndex = columnIndex;
                            orderByAscending = isAscending;
                        }
                        //int numBatches = last_table_scanned_1->getNumBatches();
                        long long totalRows = last_table_scanned_1->getNumRows();
                        
                        // std::cout << "Executing sort on column index: " << orderByColumnIndex 
                        //         << " in " << (isAscending ? "ascending" : "descending") << " order" << std::endl;
                        // std::cout << "Total rows: " << totalRows << ", Total batches: " << numBatches << std::endl;
                        
                        // Process each batch
                        // for (int batchIdx = 0; batchIdx < 1; batchIdx++) {
                        //     long long rowsInBatch;
                        //     if (batchIdx == 0) {
                        //         rowsInBatch = totalRows % BATCH_SIZE;
                        //         if (rowsInBatch == 0 && totalRows > 0) {
                        //             // If totalRows is exactly divisible by BATCH_SIZE
                        //             rowsInBatch = BATCH_SIZE;
                        //         }
                        //     } else {
                        //         rowsInBatch = BATCH_SIZE;
                        //     }
                            
                            //std::cout << "Processing batch " << batchIdx << " with " << rowsInBatch << " rows" << std::endl;
                            
                            // if (batchIdx > 0) {
                            //     //std::cout << "Loading batch " << batchIdx << " data" << std::endl;
                            //     last_table_scanned_1->getTableBatch(batchIdx);
                            // } else {
                            //     //std::cout << "Using already loaded data for batch 0" << std::endl;
                            // }
                            
                            // Sort this batch
                            if (IS_GPU) {
                                if (!ExecuteSortBatch(orderByColumnIndex, isAscending, last_table_scanned_1, last_table_scanned_1->getNumRows())) {
                                    std::cerr << "Error sorting batch " << 0 << std::endl;
                                }
                            }
                            else {
                                if (!ExecuteSortBatchCPU(orderByColumnIndex, isAscending, last_table_scanned_1, last_table_scanned_1->getNumRows())) {
                                    std::cerr << "Error sorting batch " << 0 << std::endl;
                                }
                            }
                        //}
                    }
                }
                break;
            }
            default:
                //std::cout << "Unhandeled Operator encountered:- " << PhysicalOperatorToString(execution_plan[i]->type) << std::endl;
                break;
        }
    }
}

void DuckDBManager::deleteLastTableScanned() {
    if(last_table_scanned_1 != nullptr) {
        delete last_table_scanned_1;
        last_table_scanned_1 = nullptr; 
    }
    if(last_table_scanned_2 != nullptr) {
        delete last_table_scanned_2;
        last_table_scanned_2 = nullptr; 
    }
}

DuckDBManager::~DuckDBManager() {
    deleteLastTableScanned();
}