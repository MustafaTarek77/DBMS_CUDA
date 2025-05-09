#include "db_utils_cpu.hpp"
#include "kernels.cuh"

float ExecuteMinMaxFloatCPU(int columnIdx, bool findMin, Table* last_table_scanned_h, long long numRows) {
    void* columnData = last_table_scanned_h->getData()[columnIdx];
    float* data = static_cast<float*>(columnData);
    
    float result = findMin ? FLT_MAX : -FLT_MAX;
    
    for (long long i = 0; i < numRows; i++) {
        if (std::isnan(data[i])) {
            continue;
        }
        
        
        if (findMin) {
            result = std::min(result, data[i]);
        } else {
            result = std::max(result, data[i]);
        }
    }
    
    return result;
}

long long ExecuteMinMaxDateCPU(int columnIdx, bool findMin, Table* last_table_scanned_h, long long numRows) {
    void* columnData = last_table_scanned_h->getData()[columnIdx];
    char** dateStrings = static_cast<char**>(columnData);
    
    if (numRows == 0) {
        std::cout << "No data to process for " << (findMin ? "MIN" : "MAX") << " DATE operation" << std::endl;
        return findMin ? LLONG_MAX : LLONG_MIN;
    }
    
    long long result = findMin ? LLONG_MAX : LLONG_MIN;
    
    for (long long i = 0; i < numRows; i++) {
        if (dateStrings[i] == nullptr || dateStrings[i][0] == '\0') {
            continue;
        }
        long long dateValue = dateTimeToInt64(dateStrings[i]);
        
        if ((findMin && dateValue < result) || (!findMin && dateValue > result)) {
            result = dateValue;
        }
    }
    return result;
}

// char dateStr[MAX_DATETIME];
// int64ToDateTime(result, dateStr);



double ExecuteSumFloatCPU(int columnIdx, Table* last_table_scanned_h, long long numRows) {
    void* columnData = last_table_scanned_h->getData()[columnIdx];
    float* data = static_cast<float*>(columnData);
    
    if (numRows == 0) {
        std::cout << "No data to process for SUM operation" << std::endl;
        return 0.0;
    }
    
    double sum = 0.0;
    
    for (long long i = 0; i < numRows; i++) {
        if (!std::isnan(data[i])) {
            sum += data[i];
        }
    }
    
    return sum;
}

unsigned int ExecuteCountStringCPU(int columnIdx, Table* last_table_scanned_h, long long numRows) {
    void* columnData = last_table_scanned_h->getData()[columnIdx];
    char** stringData = static_cast<char**>(columnData);
    
    if (numRows == 0) {
        std::cout << "No data to process for COUNT operation" << std::endl;
        return 0;
    }
    
    unsigned int count = 0;
    
    for (long long i = 0; i < numRows; i++) {
        if (stringData[i] != nullptr && stringData[i][0] != '\0') {
            count++;
        }
    }
    
    return count;
}

unsigned int ExecuteCountFloatCPU(int columnIdx, Table* last_table_scanned_h, long long numRows) {
    void* columnData = last_table_scanned_h->getData()[columnIdx];
    float* data = static_cast<float*>(columnData);
    
    if (numRows == 0) {
        std::cout << "No data to process for COUNT operation" << std::endl;
        return 0;
    }
    
    unsigned int count = 0;
    
    for (long long i = 0; i < numRows; i++) {
        if (!std::isnan(data[i])) {
            count++;
        }
    }
    
    return count;
}


// Merge two sorted arrays for float values
void mergeSortedArraysFloat(float* arr, int* indices, int start, int mid, int end, bool isAscending) {
    int n1 = mid - start + 1;
    int n2 = end - mid;
    
    // Create temporary arrays
    float* leftArr = new float[n1];
    int* leftIndices = new int[n1];
    float* rightArr = new float[n2];
    int* rightIndices = new int[n2];
    
    // Copy data to temporary arrays
    for (int i = 0; i < n1; i++) {
        leftArr[i] = arr[start + i];
        leftIndices[i] = indices[start + i];
    }
    for (int j = 0; j < n2; j++) {
        rightArr[j] = arr[mid + 1 + j];
        rightIndices[j] = indices[mid + 1 + j];
    }
    
    // Merge the arrays back into arr[start...end]
    int i = 0, j = 0, k = start;
    while (i < n1 && j < n2) {
        bool compare;
        
        // Handle NaN values - NaN values are always placed at the end
        bool leftIsNaN = std::isnan(leftArr[i]);
        bool rightIsNaN = std::isnan(rightArr[j]);
        
        if (leftIsNaN && rightIsNaN) {
            // Both are NaN, maintain stable sort
            compare = false;
        } else if (leftIsNaN) {
            compare = false;  // Left is NaN, so right comes first
        } else if (rightIsNaN) {
            compare = true;   // Right is NaN, so left comes first
        } else {
            // Normal comparison
            if (isAscending) {
                compare = leftArr[i] <= rightArr[j];
            } else {
                compare = leftArr[i] >= rightArr[j];
            }
        }
        
        if (compare) {
            arr[k] = leftArr[i];
            indices[k] = leftIndices[i];
            i++;
        } else {
            arr[k] = rightArr[j];
            indices[k] = rightIndices[j];
            j++;
        }
        k++;
    }
    
    // Copy the remaining elements of left array, if any
    while (i < n1) {
        arr[k] = leftArr[i];
        indices[k] = leftIndices[i];
        i++;
        k++;
    }
    
    // Copy the remaining elements of right array, if any
    while (j < n2) {
        arr[k] = rightArr[j];
        indices[k] = rightIndices[j];
        j++;
        k++;
    }
    
    // Free temporary arrays
    delete[] leftArr;
    delete[] leftIndices;
    delete[] rightArr;
    delete[] rightIndices;
}

// Merge sort for float array
void mergeSortFloat(float* arr, int* indices, int start, int end, bool isAscending) {
    if (start < end) {
        int mid = start + (end - start) / 2;
        
        // Sort first and second halves
        mergeSortFloat(arr, indices, start, mid, isAscending);
        mergeSortFloat(arr, indices, mid + 1, end, isAscending);
        
        // Merge the sorted halves
        mergeSortedArraysFloat(arr, indices, start, mid, end, isAscending);
    }
}

// Merge two sorted arrays for date values represented as long long integers
void mergeSortedArraysDate(long long* arr, int* indices, int start, int mid, int end, bool isAscending) {
    int n1 = mid - start + 1;
    int n2 = end - mid;
    
    // Create temporary arrays
    long long* leftArr = new long long[n1];
    int* leftIndices = new int[n1];
    long long* rightArr = new long long[n2];
    int* rightIndices = new int[n2];
    
    // Copy data to temporary arrays
    for (int i = 0; i < n1; i++) {
        leftArr[i] = arr[start + i];
        leftIndices[i] = indices[start + i];
    }
    for (int j = 0; j < n2; j++) {
        rightArr[j] = arr[mid + 1 + j];
        rightIndices[j] = indices[mid + 1 + j];
    }
    
    // Merge the arrays back into arr[start...end]
    int i = 0, j = 0, k = start;
    while (i < n1 && j < n2) {
        bool compare;
        
        // Handle invalid date values (typically ≤ 0)
        bool leftIsInvalid = leftArr[i] <= 0;
        bool rightIsInvalid = rightArr[j] <= 0;
        
        if (leftIsInvalid && rightIsInvalid) {
            // Both are invalid, maintain stable sort
            compare = false;
        } else if (leftIsInvalid) {
            compare = false;  // Left is invalid, so right comes first
        } else if (rightIsInvalid) {
            compare = true;   // Right is invalid, so left comes first
        } else {
            // Normal comparison
            if (isAscending) {
                compare = leftArr[i] <= rightArr[j];
            } else {
                compare = leftArr[i] >= rightArr[j];
            }
        }
        
        if (compare) {
            arr[k] = leftArr[i];
            indices[k] = leftIndices[i];
            i++;
        } else {
            arr[k] = rightArr[j];
            indices[k] = rightIndices[j];
            j++;
        }
        k++;
    }
    
    // Copy the remaining elements of left array, if any
    while (i < n1) {
        arr[k] = leftArr[i];
        indices[k] = leftIndices[i];
        i++;
        k++;
    }
    
    // Copy the remaining elements of right array, if any
    while (j < n2) {
        arr[k] = rightArr[j];
        indices[k] = rightIndices[j];
        j++;
        k++;
    }
    
    // Free temporary arrays
    delete[] leftArr;
    delete[] leftIndices;
    delete[] rightArr;
    delete[] rightIndices;
}

// Merge sort for date array
void mergeSortDate(long long* arr, int* indices, int start, int end, bool isAscending) {
    if (start < end) {
        int mid = start + (end - start) / 2;
        
        // Sort first and second halves
        mergeSortDate(arr, indices, start, mid, isAscending);
        mergeSortDate(arr, indices, mid + 1, end, isAscending);
        
        // Merge the sorted halves
        mergeSortedArraysDate(arr, indices, start, mid, end, isAscending);
    }
}

// Merge two sorted arrays for string values
void mergeSortedArraysString(char** arr, int* indices, int start, int mid, int end, bool isAscending) {
    int n1 = mid - start + 1;
    int n2 = end - mid;
    
    // Create temporary arrays
    char** leftArr = new char*[n1];
    int* leftIndices = new int[n1];
    char** rightArr = new char*[n2];
    int* rightIndices = new int[n2];
    
    // Copy data to temporary arrays
    for (int i = 0; i < n1; i++) {
        leftArr[i] = arr[start + i];
        leftIndices[i] = indices[start + i];
    }
    for (int j = 0; j < n2; j++) {
        rightArr[j] = arr[mid + 1 + j];
        rightIndices[j] = indices[mid + 1 + j];
    }
    
    // Merge the arrays back into arr[start...end]
    int i = 0, j = 0, k = start;
    while (i < n1 && j < n2) {
        bool compare;
        
        // Handle null strings
        bool leftIsNull = (leftArr[i] == nullptr);
        bool rightIsNull = (rightArr[j] == nullptr);
        
        if (leftIsNull && rightIsNull) {
            // Both are null, maintain stable sort
            compare = false;
        } else if (leftIsNull) {
            compare = false;  // Left is null, so right comes first
        } else if (rightIsNull) {
            compare = true;   // Right is null, so left comes first
        } else {
            // Normal string comparison
            int cmpResult = strcmp(leftArr[i], rightArr[j]);
            if (isAscending) {
                compare = (cmpResult <= 0);
            } else {
                compare = (cmpResult >= 0);
            }
        }
        
        if (compare) {
            arr[k] = leftArr[i];
            indices[k] = leftIndices[i];
            i++;
        } else {
            arr[k] = rightArr[j];
            indices[k] = rightIndices[j];
            j++;
        }
        k++;
    }
    
    // Copy the remaining elements of left array, if any
    while (i < n1) {
        arr[k] = leftArr[i];
        indices[k] = leftIndices[i];
        i++;
        k++;
    }
    
    // Copy the remaining elements of right array, if any
    while (j < n2) {
        arr[k] = rightArr[j];
        indices[k] = rightIndices[j];
        j++;
        k++;
    }
    
    // Free temporary arrays
    delete[] leftArr;
    delete[] leftIndices;
    delete[] rightArr;
    delete[] rightIndices;
}

// Merge sort for string array
void mergeSortString(char** arr, int* indices, int start, int end, bool isAscending) {
    if (start < end) {
        int mid = start + (end - start) / 2;
        
        // Sort first and second halves
        mergeSortString(arr, indices, start, mid, isAscending);
        mergeSortString(arr, indices, mid + 1, end, isAscending);
        
        // Merge the sorted halves
        mergeSortedArraysString(arr, indices, start, mid, end, isAscending);
    }
}

bool ExecuteSortBatchCPU(int columnIdx, bool isAscending, Table* table, long long rowsInBatch) {
    // Get data from the table
    void** tableData = table->getData();
    int numColumns = table->getNumColumns();
    char** columnNames = table->getColumnNames();
    
    // Parse column types from column names
    std::vector<char> columnTypes(numColumns);
    for (int i = 0; i < numColumns; i++) {
        if (columnNames[i]) {
            std::string colName = columnNames[i];
            size_t typePos = colName.rfind('(');
            if (typePos != std::string::npos && colName.length() > typePos + 1) {
                columnTypes[i] = colName[typePos + 1]; 
            } else {
                columnTypes[i] = 'T'; // Default to text
            }
        } else {
            columnTypes[i] = 'T'; // Default to text
        }
    }
    
    // Determine the column type for sorting
    char columnType = columnTypes[columnIdx];
    
    // Allocate indices array
    int* indices = new int[rowsInBatch];
    
    // Initialize indices
    for (int i = 0; i < rowsInBatch; i++) {
        indices[i] = i;
    }
    
    // Handle based on column type
    if (columnType == 'D') { // Date/DateTime column
        // Convert dates to integers
        long long* date_ints = new long long[rowsInBatch];
        char** dateStrings = static_cast<char**>(tableData[columnIdx]);
        
        for (long long i = 0; i < rowsInBatch; i++) {
            date_ints[i] = dateTimeToInt64(dateStrings[i]);
        }
        
        // Sort date values using merge sort
        mergeSortDate(date_ints, indices, 0, rowsInBatch - 1, isAscending);
        
        // Create backups of all columns
        void** backupData = new void*[numColumns];
        
        for (int col = 0; col < numColumns; col++) {
            char colType = columnTypes[col];
            
            switch (colType) {
                case 'N': // Numeric
                    backupData[col] = malloc(rowsInBatch * sizeof(float));
                    memcpy(backupData[col], tableData[col], rowsInBatch * sizeof(float));
                    break;
                    
                case 'T': // Text
                case 'D': // Date
                default:  // Default to string
                    backupData[col] = malloc(rowsInBatch * sizeof(char*));
                    memcpy(backupData[col], tableData[col], rowsInBatch * sizeof(char*));
                    break;
            }
        }
        
        // Reorder all columns using sorted indices
        for (int col = 0; col < numColumns; col++) {
            char colType = columnTypes[col];
            
            switch (colType) {
                case 'N': // Numeric (float)
                    {
                        float* typedColumnData = static_cast<float*>(tableData[col]);
                        float* typedBackupData = static_cast<float*>(backupData[col]);
                        
                        // Reorder based on sorted indices
                        for (int i = 0; i < rowsInBatch; i++) {
                            typedColumnData[i] = typedBackupData[indices[i]];
                        }
                    }
                    break;
                    
                case 'T': // Text
                case 'D': // Date
                default:  // Default to string
                    {
                        char** typedColumnData = static_cast<char**>(tableData[col]);
                        char** typedBackupData = static_cast<char**>(backupData[col]);
                        
                        // Reorder based on sorted indices
                        for (int i = 0; i < rowsInBatch; i++) {
                            typedColumnData[i] = typedBackupData[indices[i]];
                        }
                    }
                    break;
            }
        }
        
        // Free resources
        delete[] date_ints;
        for (int col = 0; col < numColumns; col++) {
            free(backupData[col]);
        }
        delete[] backupData;
    }
    else if (columnType == 'N') { // Regular numeric column (float)
        // Copy the float values
        float* values = new float[rowsInBatch];
        float* sortColumn = static_cast<float*>(tableData[columnIdx]);
        
        // Copy values to temporary array
        memcpy(values, sortColumn, rowsInBatch * sizeof(float));
        
        // Sort float values using merge sort
        mergeSortFloat(values, indices, 0, rowsInBatch - 1, isAscending);
        
        // Create backups of all columns
        void** backupData = new void*[numColumns];
        
        for (int col = 0; col < numColumns; col++) {
            char colType = columnTypes[col];
            
            switch (colType) {
                case 'N': // Numeric
                    backupData[col] = malloc(rowsInBatch * sizeof(float));
                    memcpy(backupData[col], tableData[col], rowsInBatch * sizeof(float));
                    break;
                    
                case 'T': // Text
                case 'D': // Date
                default:  // Default to string
                    backupData[col] = malloc(rowsInBatch * sizeof(char*));
                    memcpy(backupData[col], tableData[col], rowsInBatch * sizeof(char*));
                    break;
            }
        }
        
        // Reorder all columns using sorted indices
        for (int col = 0; col < numColumns; col++) {
            char colType = columnTypes[col];
            
            switch (colType) {
                case 'N': // Numeric (float)
                    {
                        float* typedColumnData = static_cast<float*>(tableData[col]);
                        float* typedBackupData = static_cast<float*>(backupData[col]);
                        
                        // Reorder based on sorted indices
                        for (int i = 0; i < rowsInBatch; i++) {
                            typedColumnData[i] = typedBackupData[indices[i]];
                        }
                    }
                    break;
                    
                case 'T': // Text
                case 'D': // Date
                default:  // Default to string
                    {
                        char** typedColumnData = static_cast<char**>(tableData[col]);
                        char** typedBackupData = static_cast<char**>(backupData[col]);
                        
                        // Reorder based on sorted indices
                        for (int i = 0; i < rowsInBatch; i++) {
                            typedColumnData[i] = typedBackupData[indices[i]];
                        }
                    }
                    break;
            }
        }
        
        // Free resources
        delete[] values;
        for (int col = 0; col < numColumns; col++) {
            free(backupData[col]);
        }
        delete[] backupData;
    }
    else if (columnType == 'T') { // Text column
        // Use the original string pointers for sorting
        char** textColumn = static_cast<char**>(tableData[columnIdx]);
        
        // Create a copy of string pointers for the merge sort
        char** textCopy = new char*[rowsInBatch];
        memcpy(textCopy, textColumn, rowsInBatch * sizeof(char*));
        
        // Sort string values using merge sort
        mergeSortString(textCopy, indices, 0, rowsInBatch - 1, isAscending);
        
        // Create backups of all columns
        void** backupData = new void*[numColumns];
        
        for (int col = 0; col < numColumns; col++) {
            char colType = columnTypes[col];
            
            switch (colType) {
                case 'N': // Numeric
                    backupData[col] = malloc(rowsInBatch * sizeof(float));
                    memcpy(backupData[col], tableData[col], rowsInBatch * sizeof(float));
                    break;
                    
                case 'T': // Text
                case 'D': // Date
                default:  // Default to string
                    backupData[col] = malloc(rowsInBatch * sizeof(char*));
                    memcpy(backupData[col], tableData[col], rowsInBatch * sizeof(char*));
                    break;
            }
        }
        
        // Reorder all columns using sorted indices
        for (int col = 0; col < numColumns; col++) {
            char colType = columnTypes[col];
            
            switch (colType) {
                case 'N': // Numeric (float)
                    {
                        float* typedColumnData = static_cast<float*>(tableData[col]);
                        float* typedBackupData = static_cast<float*>(backupData[col]);
                        
                        // Reorder based on sorted indices
                        for (int i = 0; i < rowsInBatch; i++) {
                            typedColumnData[i] = typedBackupData[indices[i]];
                        }
                    }
                    break;
                    
                case 'T': // Text
                case 'D': // Date
                default:  // Default to string
                    {
                        char** typedColumnData = static_cast<char**>(tableData[col]);
                        char** typedBackupData = static_cast<char**>(backupData[col]);
                        
                        // Reorder based on sorted indices
                        for (int i = 0; i < rowsInBatch; i++) {
                            typedColumnData[i] = typedBackupData[indices[i]];
                        }
                    }
                    break;
            }
        }
        
        // Free resources
        delete[] textCopy;
        for (int col = 0; col < numColumns; col++) {
            free(backupData[col]);
        }
        delete[] backupData;
    }
    
    // Free indices
    delete[] indices;
    
    return true;
}


// CPU versions of the comparison functions
bool compareFloatsCPU(float lhs, const char& op, float rhs) {
    if (op == '=') return lhs == rhs;
    if (op == '!') return lhs != rhs;
    if (op == '<') return lhs < rhs;
    if (op == '>') return lhs > rhs;
    return false;
}

bool compareStringsCPU(const char* lhs, const char& op, const char* rhs) {
    if (lhs == nullptr || rhs == nullptr) return false;
    
    // For strings in the CPU version, we need to use strcmp for content comparison
    // not just pointer comparison like in the GPU version
    int cmp = strcmp(lhs, rhs);
    
    if (op == '=') return cmp == 0;
    if (op == '!') return cmp != 0;
    if (op == '<') return cmp < 0;
    if (op == '>') return cmp > 0;
    return false;
}

int parseDateTimeCPU(const char* datetime) {
    if (datetime == nullptr) return 0;
    
    // Handle optional ::TIMESTAMP suffix
    int len = strlen(datetime);

    // Find start of actual timestamp
    const char* ptr = datetime;

    // If the string is quoted, skip the quotes
    if (*ptr == '\'') ++ptr;

    // Check if the datetime ends with '::TIMESTAMP'
    const char* ts_suffix = "::TIMESTAMP";
    int suffix_len = 11;
    if (len >= suffix_len) {
        bool has_suffix = true;
        for (int i = 0; i < suffix_len; ++i) {
            if (datetime[len - suffix_len + i] != ts_suffix[i]) {
                has_suffix = false;
                break;
            }
        }
        if (has_suffix) {
            // Make a null-terminated copy without the suffix for parsing
            char clean[20];
            for (int i = 0; i < 19 && ptr[i] != '\0'; ++i) {
                clean[i] = ptr[i];
                clean[i + 1] = '\0';
            }
            return parseDateTimeCPU(clean); // Recursively parse the cleaned string
        }
    }

    // Parse the date format: "YYYY-MM-DD HH:MM:SS"
    if (len < 19) return 0; // Not enough characters
    
    int year = 0, month = 0, day = 0, hour = 0, min = 0, sec = 0;

    // Ensure the characters are numerics before parsing
    if (!isdigit(datetime[0]) || !isdigit(datetime[1]) || !isdigit(datetime[2]) || !isdigit(datetime[3]) ||
        !isdigit(datetime[5]) || !isdigit(datetime[6]) ||
        !isdigit(datetime[8]) || !isdigit(datetime[9]) ||
        !isdigit(datetime[11]) || !isdigit(datetime[12]) ||
        !isdigit(datetime[14]) || !isdigit(datetime[15]) ||
        !isdigit(datetime[17]) || !isdigit(datetime[18])) {
        return 0;
    }

    year  = (datetime[0]-'0')*1000 + (datetime[1]-'0')*100 + (datetime[2]-'0')*10 + (datetime[3]-'0');
    month = (datetime[5]-'0')*10 + (datetime[6]-'0');
    day   = (datetime[8]-'0')*10 + (datetime[9]-'0');
    hour  = (datetime[11]-'0')*10 + (datetime[12]-'0');
    min   = (datetime[14]-'0')*10 + (datetime[15]-'0');
    sec   = (datetime[17]-'0')*10 + (datetime[18]-'0');

    // Simple conversion to seconds since a base date 
    return (((year * 12 + month) * 31 + day) * 24 + hour) * 60 * 60 + min * 60 + sec;
}

bool compareDatesCPU(const char* lhs, const char& op, const char* rhs) {
    if (lhs == nullptr || rhs == nullptr) return false;
    
    int lhs_time = parseDateTimeCPU(lhs);
    int rhs_time = parseDateTimeCPU(rhs);

    if (op == '=') return lhs_time == rhs_time;
    if (op == '!') return lhs_time != rhs_time;
    if (op == '<') return lhs_time < rhs_time;
    if (op == '>') return lhs_time > rhs_time;
    return false;
}

void ExecuteJoinCPU(Kernel_Condition** conditions, int num_total_conditions, 
                 const std::vector<int>& conditions_groups_sizes, 
                 const std::vector<std::string>& projections, 
                 Table* last_table_scanned_1, Table* last_table_scanned_2, 
                 int*& h_out1, int*& h_out2, int& h_result_count) {  
    int num_groups_conditions = conditions_groups_sizes.size();
    void** h_data1 = last_table_scanned_1->getData();
    void** h_data2 = last_table_scanned_2->getData();

    int rows1 = last_table_scanned_1->getNumRows();
    int rows2 = last_table_scanned_2->getNumRows();
    int totalPairs = rows1 * rows2;
    
    // Allocate output arrays
    try {
        h_out1 = new int[totalPairs];
        h_out2 = new int[totalPairs];
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation error: " << e.what() << std::endl;
        h_result_count = 0;
        return;
    }
    
    // Flatten conditions for easier processing
    Kernel_Condition* flat_conditions = nullptr;
    int* group_offsets = nullptr;
    int* group_sizes = nullptr;
    
    try {
        flat_conditions = new Kernel_Condition[num_total_conditions];
        group_offsets = new int[num_groups_conditions]; // where each group starts
        group_sizes = new int[num_groups_conditions];   // size of each group
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation error: " << e.what() << std::endl;
        delete[] h_out1;
        delete[] h_out2;
        h_out1 = nullptr;
        h_out2 = nullptr;
        h_result_count = 0;
        return;
    }

    int flat_index = 0;
    for (int g = 0; g < num_groups_conditions; ++g) {
        group_offsets[g] = flat_index;
        int group_size = 0;
        for (int i = 0; i < conditions_groups_sizes[g]; ++i) { 
            flat_conditions[flat_index++] = conditions[g][i];
            ++group_size;
        }
        group_sizes[g] = group_size;
    }
    
    // Initialize result count
    h_result_count = 0;
    
    // Main join loop - check all row pairs
    for (int idx = 0; idx < totalPairs; ++idx) {
        int row1 = idx / rows2;
        int row2 = idx % rows2;
        
        bool match_found = false;
        if (num_groups_conditions > 0) {
            for (int g = 0; g < num_groups_conditions && !match_found; ++g) {
                int group_start = group_offsets[g];
                int group_size = group_sizes[g];
                bool group_ok = true;

                for (int c = 0; c < group_size && group_ok; ++c) {
                    Kernel_Condition cond = flat_conditions[group_start + c];

                    if (cond.type == 'N') {
                        float val1 = ((float*)h_data1[cond.idx1])[row1];
                        float val2 = 0;
                        if (cond.idx2 == -1) {
                            val2 = *((float*)cond.value);
                        }
                        else {
                            val2 = ((float*)h_data2[cond.idx2])[row2];
                        }
                        if (!compareFloatsCPU(val1, cond.relational_operator, val2)) {
                            group_ok = false;
                        }
                    }
                    else if (cond.type == 'T') {
                        char* val1 = ((char**)h_data1[cond.idx1])[row1];
                        char* val2;
                        if (cond.idx2 == -1) {
                            val2 = (char*)cond.value;
                        }
                        else {
                            val2 = ((char**)h_data2[cond.idx2])[row2];
                        }
                        if (!compareStringsCPU(val1, cond.relational_operator, val2)) {
                            group_ok = false;
                        }
                    }
                    else if (cond.type == 'D') {
                        char* val1 = ((char**)h_data1[cond.idx1])[row1];
                        char* val2;
                        if (cond.idx2 == -1) {
                            val2 = (char*)cond.value;
                        }
                        else {
                            val2 = ((char**)h_data2[cond.idx2])[row2];
                        }
                        if (!compareDatesCPU(val1, cond.relational_operator, val2)) {
                            group_ok = false;
                        }
                    }
                }

                if (group_ok) {
                    match_found = true;
                }
            }

            if (match_found) {
                h_out1[h_result_count] = row1;
                h_out2[h_result_count] = row2;
                h_result_count++;
            }
        }
        else {
            // No conditions - Cartesian product
            h_out1[h_result_count] = row1;
            h_out2[h_result_count] = row2;
            h_result_count++;
        }
    }
    
    // Clean up
    delete[] flat_conditions;
    delete[] group_offsets;
    delete[] group_sizes;
}