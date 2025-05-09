## Setup Instructions

### 1. Clone DuckDB

Clone the DuckDB repository inside your project directory:

```bash
git clone https://github.com/duckdb/duckdb.git
```

### 2. Build DuckDB

Navigate to the DuckDB directory and build it:

```bash
cd duckdb
make
```

### 3. Compile the Project

Use `nvcc` to compile the project with CUDA support. Replace `/path/to/your/project` with the actual path to your DuckDB folder.

```bash
nvcc main.cpp DB.cpp Table.cpp db_utils.cu db_utils_cpu.cpp kernels.cu utils.cpp -o main \
-I/path/to/your/project/duckdb/src/include \
-L/path/to/your/project/duckdb/build/release/src \
-lduckdb -std=c++17 -Wno-deprecated-gpu-targets
```

### 4. Set Library Path

Before running the program, add the DuckDB library path to your environment:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/project/duckdb/build/release/src
```

### 5. Prepare Your Query

Write your SQL query in a plain `.txt` file.

### 6. Run the Program

To run your program, use the following command:

```bash
./main ./Data /path/to/your/query.txt
```

Replace `/path/to/your/query.txt` with the path to your SQL query file.

