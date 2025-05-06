#include "./DB.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
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
#include "db_utils.hpp"

namespace fs = std::filesystem;
using namespace duckdb;

DuckDBManager::DuckDBManager(const std::string &csv_directory)
    : csv_directory(csv_directory), db(std::make_unique<DuckDB>(nullptr)), con(std::make_unique<Connection>(*db)) {}

void DuckDBManager::InitializeDatabase() {
    con->Query("SET disabled_optimizers = 'filter_pushdown, statistics_propagation';");
    // Collect all table names
    for (const auto &entry : fs::directory_iterator(csv_directory)) {
        if (entry.path().extension() == ".csv") {
            table_names.push_back(entry.path().stem().string());
        }
    }
}

void DuckDBManager::LoadTablesFromCSV() {
    for (const auto &entry : fs::directory_iterator(csv_directory)) {
        if (entry.path().extension() == ".csv") {
            std::string csv_file = entry.path().string();
            std::string table_name = entry.path().stem().string();

            auto columns = GetCSVHeaders(csv_file, table_names);
            std::string create_query = ConstructCreateTableQuery(columns, table_name);

            std::cout << "Executing: " << create_query << std::endl;
            con->Query(create_query);            
        }
    }
}

std::vector<ColumnInfo> DuckDBManager::GetCSVHeaders(const std::string &csv_file, const std::vector<std::string> &table_names) {
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
                info.name = header.substr(0, header.find("(P)"));
                info.is_primary_key = true;
            }

            if (header.find("(N)") != std::string::npos) {
                info.type = "FLOAT"; 
                info.name = header.substr(0, header.find("(N)"));
            } else if (header.find("(T)") != std::string::npos) {
                info.type = "VARCHAR";
                info.name = header.substr(0, header.find("(T)"));
            } else if (header.find("(D)") != std::string::npos) {
                info.type = "TIMESTAMP";
                info.name = header.substr(0, header.find("(D)"));
            } else {
                info.type = "VARCHAR";
                info.name = header;
            }

            for (const auto &table_name : table_names) {
                if (header.find(table_name + "_") != std::string::npos) {
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
            std::string referenced_column = key.substr(pos + 1);

            query += ", FOREIGN KEY (" + primary_key_column + ") REFERENCES " + referenced_table + " (" + referenced_column + ")";
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
    
    std::cout << "=== PHYSICAL PLAN ===" << std::endl;
    
    std::cout << physical_plan.get()->Root().ToString() << std::endl;   
    TraversePlan(&physical_plan.get()->Root());
    ExecutePlan();
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
                deleteLastTableScanned(); 

                auto params = execution_plan[i]->ParamsToString();
                std::string table_name;
                std::vector<std::string> projections, target_columns;
                std::vector<std::vector<Condition>> conditions;

                // Getting table name and initial projections
                for(auto &param: params)
                {
                    if(param.first=="Table") {
                        std::cout<<"Scanning Table:- "<<param.second<<std::endl;
                        table_name = param.second;
                    }
                    else if(param.first=="Projections") {
                        std::cout << "Projections: ";
                        std::istringstream iss(param.second);
                        std::string projection;
                        while (std::getline(iss, projection)) {
                            if (!projection.empty()) {
                                std::cout << projection << " ";
                                projections.push_back(projection);
                            }
                        }
                        std::cout << std::endl;
                    }
                }
                target_columns = projections;
                
                // Getting Conditions if exist
                if(i+1<execution_plan.size() && execution_plan[i+1]->type==PhysicalOperatorType::FILTER) {
                    auto params = execution_plan[i+1]->ParamsToString();
                    std::cout << "Filters:-" << std::endl;
                
                    for (auto &param : params) {
                        if (param.first=="__expression__") {
                            std::string filter_str = param.second;
                            std::cout << "Processing filter string: "<< filter_str << std::endl;
                            parseComplexExpression(filter_str, conditions);
                            // printConditionStructure(conditions);
                        }
                    }
                    i++;
                }

                // Getting Projections if exist
                if(i+1<execution_plan.size() && execution_plan[i+1]->type==PhysicalOperatorType::PROJECTION) {
                    target_columns.clear();
                    auto params = execution_plan[i+1]->ParamsToString();
                    std::cout << "Projecting expressions:- ";
                    for(auto &param: params) {
                        if (param.first=="__projections__") {
                            std::cout<< param.second<<std::endl;
                            std::istringstream iss(param.second);
                            std::string projection;
                            while (std::getline(iss, projection)) {
                                if (!projection.empty()) {
                                    if(projection[0]=='#') {
                                        std::string proj = projection.substr(1);
                                        std::cout << projections[std::stoi(proj)] << " ";
                                        target_columns.push_back(projections[std::stoi(proj)]);
                                    }
                                    else {
                                        std::cout << projection << " ";
                                        target_columns.push_back(projection);
                                    }
                                }
                            }
                        }
                    }
                    std::cout << std::endl;
                    i++;
                }

                last_table_scanned_h = new Table(table_name, projections, target_columns, conditions);
                last_table_scanned_h->printData();
                std::cout<<"Table is scanned successfully"<<std::endl;
                break;
            }
            case PhysicalOperatorType::FILTER: {
                auto params = execution_plan[i]->ParamsToString();
                for(auto &param: params)
                {
                    std::cout<<"Applying Filter:- "<<param.second<<std::endl;
                }
                break;
            }
            case PhysicalOperatorType::PROJECTION: {
                //auto &proj = execution_plan[i]->Cast<PhysicalProjection>();
                std::cout << "Projecting hhh:- ";
                // for (auto &expr : proj.select_list) {
                //     std::cout << expr->ToString() << " ";
                // }
                std::cout << std::endl;
                break;
            }
            case PhysicalOperatorType::HASH_JOIN: {
                // auto &join = v->Cast<PhysicalHashJoin>();
                // std::cout << "Hash Join on condition:- " << join.conditions[0].left->ToString() 
                //           << " = " << join.conditions[0].right->ToString() << std::endl;
                break;
            }
            case PhysicalOperatorType::UNGROUPED_AGGREGATE: {
                auto params = execution_plan[i]->ParamsToString();
                std::cout << "Applying Aggregation:-" << std::endl;
                
                std::vector<std::string> agg_functions;
                std::vector<std::string> agg_targets;
                
                for (auto &param : params) {
                    if (param.first == "Aggregates") {
                        std::cout << "Aggregate Functions: " << std::endl;

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

                                std::cout << "  Function: " << function << ", Column Index: " << columnIdx << std::endl;
                                
                                ExecuteAggregateFunction(function, columnIdx, last_table_scanned_h);
                            }
                        }
                    }
                }
                break;
            }
            case PhysicalOperatorType::CROSS_PRODUCT: {
                std::cout << "Cross Product:-" << std::endl;
                break;
            }
            case PhysicalOperatorType::ORDER_BY: {
                // auto &order = op->Cast<PhysicalOrder>();
                std::cout << "Ordering by:- ";
                // for (auto &expr : order.orders) {
                //     std::cout << expr.expression->ToString() << " ";
                //     std::cout << (expr.type == OrderType::ASCENDING ? "ASC" : "DESC") << " ";
                // }
                std::cout << std::endl;
                break;
            }
            default:
                std::cout << "Unhandeled Operator encountered:- " << PhysicalOperatorToString(execution_plan[i]->type) << std::endl;
                break;
        }
    }
}

void DuckDBManager::deleteLastTableScanned() {
    delete last_table_scanned_h;
    last_table_scanned_h = nullptr; 
}

DuckDBManager::~DuckDBManager() {
    deleteLastTableScanned();
}