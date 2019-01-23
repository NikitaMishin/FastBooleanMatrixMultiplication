
#include <sstream>
#include <fstream>
#include <set>
#include <iostream>
#include <map>
#include <vector>
#include <iostream>
#include <bits/stdc++.h>
#include <sstream>
#include <ctime>
#include <cstdint>
#include <stdint.h>


#define SQUEEZE 32



#define DELIMITR " "
#define IS_CHANGED 1
#define NOT_CHANGED 0

#define  BLOCK_SIZE 32

using namespace std;
using String = std::string;



#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#define CHANGED 1
#define NOT_CHANGED 0
#define lsb(i) ((i) & -(i)) // return least significant bit
#define BITS sizeof(__uint32_t) * 8// aka 32


uint32_t ** allocate_tables(int num_tables,int num_rows,int num_cols){

    uint32_t ** d_ppcPtr, *d_pcPtr;
    cudaMalloc(&d_ppcPtr, sizeof(uint32_t*) * num_tables);


    for(int i = 0; i < num_tables; i ++)
    {
        cudaMalloc(&d_pcPtr, sizeof(uint32_t) * num_rows *num_cols );
        cudaMemset(d_pcPtr, 0, sizeof(uint32_t) * num_rows *num_cols );
        cudaMemcpy(&d_ppcPtr[i], &d_pcPtr, sizeof(uint32_t*), cudaMemcpyHostToDevice);
    }
    return d_ppcPtr;
}

void delete_tables(uint32_t**tables,int num_tables){
    uint32_t *d_pcPtr;
    for(int i = 0; i < num_tables; i ++)
    {
        //cudaMemcpy(&d_pcPtr,&tables[i], sizeof(uint32_t*), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_pcPtr,&tables[i], sizeof(uint32_t*), cudaMemcpyDeviceToHost);
        cudaFree(d_pcPtr);
    }
    cudaFree(tables);
}




//return the next number with the same number of bits
__device__ int snoob(int i)
{
    int least = lsb(i);
    int ripple = i + least;
    return (((ripple ^ i) >> 2) / least) | ripple;
}


#define K 8
#define BLOCK_SIZE 32
/*
*  For C=AXB  make part of  table
*  lookup_tables part table
*  cols - cols for current  cols size of part of table
*  rows - rows in A,B,C
*  tables_num  in nums of tables
*  real_cols cols in A,B,C
*  offset - for what part of table
*/
__global__ void make_table(uint32_t *B, uint32_t ** lookup_tables, int cols, int rows, int tables_num, int real_cols, int offset) {

//каждый элемент считает часть таблицу (256 элементов)   8 элементов в матр-це б
    int x_col = blockIdx.x * BLOCK_SIZE + threadIdx.x; // позиция в колонке этого потока
    int y_row = (blockIdx.y * BLOCK_SIZE + threadIdx.y)*K; // позиция в строке в таблице B самая верхняя
    int twokey = (1<<K);
    int i;
    int least,rest;


    if(x_col >= cols || y_row >= rows ) {
// если поток выходит за рамки нашей таблицы по ширине, то ему ничего не надо делать
        return;
    }

    uint32_t *T = lookup_tables[blockIdx.y * BLOCK_SIZE + threadIdx.y];  //pointer to table offset for case when too large num tables


    T[x_col] = 0; // row with 000000000...


// fill when 1 bit
#pragma unroll
    for(int j = 0; j < K; j++) {
        i = 1<<(j);
        T[i * cols + x_col] = B[ (y_row + j) * real_cols  + x_col + offset];
    }//kk

// fill when 2 and so on...
#pragma unroll
    for(int h = 2;h <= K; h++) {
//  iterate all integers with h bits, and < 2^k
        i = (1 << h) - 1;
        for (;i < twokey; i = snoob(i)) {
            least = lsb(i);
            rest = i - least;
//T[least] and T[rest] already calculated
            T[i * cols + x_col ] = T[ least* cols + x_col] | T[ rest*cols + x_col];
        }
    }
}


#define BLOCK_SIZE_COL 32
#define BLOCK_SIZE_ROW 32

__device__ int get_actual_key(uint32_t composite_key,int j){
    return  (0xFF) & (composite_key >> (8 * j));
}



__device__ uint32_t is_changed_matrix = 0;


/*
* ВАЖНО: Если C  и A одна и  та же матрица, необходимо заменить одну из них копией, а потом поинетр перекинуть
*  For C=AXB  perform multiplication over subring F2
*  lookup_tables part table
*  rows - rows in A,B,C
*  cols - cols in A,B,C
*  full_steps  nums of steps
*  small_steps small_steps
*  offset - for what part of C we calculate
*  is_changed - flag that determine wheather matrix changed during multiplication
*
*  настройка ядер для умножения
*  dim3 dimBlock_m4ri(BLOCK_SIZE_COL,BLOCK_SIZE_ROW);
*
*  настройка для умножения из полного цикла
*  grid_x = table_cols_n /BLOCK_SIZE_COL;
*  if(table_cols_n%BLOCK_SIZE_COL!=0)grid_x++;
*  grid_y = rows/BLOCK_SIZE_ROW;
*  if(rows%BLOCK_SIZE_ROW!=0)grid_y++;
*  dim3 dimGrid_m4ri_nums(grid_x,grid_y);
*
*  настройка для последнего умножения
*  grid_x = table_cols_last/BLOCK_SIZE_COL;
*  if(table_cols_last%BLOCK_SIZE_COL!=0)grid_x++;
*  dim3 dimGrid_m4ri_last(grid_x,grid_y);
*
*  Пример
*    // full_step = table_cols_n/BLOCK_SIZE_COL;
*    // small_step = table_cols_n%BLOCK_SIZE_COL;
*    // offset = i*table_cols_n
*    for(int i = 0;i < num_launches;i++){
*        make_table<<<dimGrid_table_nums,dimBlock_table_kernel>>>(B,tables_n,table_cols_n,rows,num_tables,cols,i*table_cols_n);
*        cudaDeviceSynchronize();
*        m4ri_mul<<<dimGrid_m4ri_nums,dimBlock_m4ri>>>(A,C,tables_n,rows,cols,table_cols_n,table_cols_n/BLOCK_SIZE_COL,table_cols_n%BLOCK_SIZE_COL,i*table_cols_n,is_changed_device);
*        cudaDeviceSynchronize();
*    }
*    // full_step = table_cols_last/BLOCK_SIZE_COL;
*    // small_step = table_cols_last%BLOCK_SIZE_COL;
*    // offset = num_launches*table_cols_n
*    if(table_cols_last != 0){
*
*        make_table<<<dimGrid_table_last,dimBlock_table_kernel>>>(B,tables_last,table_cols_last,rows,num_tables,cols,num_launches*table_cols_n);
*        cudaDeviceSynchronize();
*        m4ri_mul<<<dimGrid_m4ri_last,dimBlock_m4ri>>>(A,C,tables_last,rows,cols,table_cols_last,table_cols_last/BLOCK_SIZE_COL,table_cols_last%BLOCK_SIZE_COL,num_launches*table_cols_n,is_changed_device);
*        cudaDeviceSynchronize();
*    }
*
*/
__global__ void m4ri_mul(uint32_t *A, uint32_t *C, uint32_t **lookup_tables,int rows, int cols,int cols_table,int full_steps,int small_step,int offset) {
// каждый поток заполняет 1 элемент в в матрице С
__shared__ uint32_t local_A[BLOCK_SIZE_ROW][BLOCK_SIZE_COL];
int col_x = threadIdx.x + blockIdx.x * BLOCK_SIZE_COL + offset; // where in  C
int row_y = threadIdx.y + blockIdx.y * BLOCK_SIZE_ROW; // where in C
int last = cols % BLOCK_SIZE_COL;// определяет сколько при неполном надо ключей набирать

int col_in_T = threadIdx.x + blockIdx.x * BLOCK_SIZE_COL;// по совместительству сколько эл-ов максимльно мы сейчас можем обработать

uint32_t *T;
uint32_t composite_key;
int actual_key;
uint32_t oldC;

if(col_x < cols && col_in_T < cols_table && row_y < rows) {
oldC = C[row_y * cols + col_x];
} else {
oldC = 0;
}



uint32_t tmp;
uint32_t value = 0;
#pragma unroll
for(int i = 0; i < full_steps; i++) {

// все полные прогоны по ключам
tmp = __brev(A[ row_y * cols + threadIdx.x + i * BLOCK_SIZE_COL]); // reverse
local_A[threadIdx.y][threadIdx.x] = tmp;
__syncthreads();



for(int t = 0; t < BLOCK_SIZE_COL; t++) {
composite_key = local_A[threadIdx.y][t];
for(int j = 0; j < 4;j++) {
T = lookup_tables[BLOCK_SIZE_COL * i*4 + t*4 + j];
actual_key = get_actual_key(composite_key,j);
value |= T[actual_key * cols_table + col_in_T];//add if вроде не надо

}
}
}

__syncthreads();

if(small_step) {
int cur_step = full_steps;
if(threadIdx.x + cur_step * BLOCK_SIZE_COL < cols  && row_y < rows){
tmp = __brev(A[ row_y * cols + threadIdx.x + cur_step * BLOCK_SIZE_COL]); // reverse
local_A[threadIdx.y][threadIdx.x] = tmp;
}
__syncthreads();
//потоки которые выхлжят им нечего делать, свой вклад в загрузку они уже внесли
if(col_x >= cols || col_in_T >= cols_table  || row_y >= rows) {
return;
}

for(int t = 0; t < last; t++) {
composite_key = local_A[threadIdx.y][t];
for(int j = 0; j < 4;j++) {
T = lookup_tables[cur_step * BLOCK_SIZE_COL * 4 + t*4 + j];
actual_key = get_actual_key(composite_key,j);
value |= T[actual_key * cols_table + col_in_T];
}
}
}
value = value|oldC;

if(is_changed_matrix == NOT_CHANGED && value!=oldC){
is_changed_matrix = CHANGED;
}

if(col_x < cols && row_y < rows && col_in_T < cols_table && value != oldC) {
C[row_y * cols + col_x] = oldC | value;
}

}


uint32_t * allocate_matrix_device(int rows,int cols){
    uint32_t *matrix;
    cudaMalloc((void **) &matrix, sizeof(uint32_t)*rows*cols);
    return matrix;

}

uint32_t * allocate_matrix_host(int rows,int cols) {
    // allocate memory in host RAM
    uint32_t *matrix;
    cudaMallocHost((void **) &matrix, sizeof(uint32_t)*rows * cols);
    return matrix;
}


// a =cb и таков порядок аргубемнов
int wrapper_m4ri(uint32_t *a,uint32_t *c,uint32_t *b,int rows,int cols){
int table_cols_max = cols;
int num_tables = rows/K;
int num_launches = cols/table_cols_max;
int table_cols_n = table_cols_max;
int table_cols_last =  cols % table_cols_max;
uint32_t * a_d  = allocate_matrix_device(rows,cols);
uint32_t * b_d  = allocate_matrix_device(rows,cols);
uint32_t * c_d  = allocate_matrix_device(rows,cols);

cudaMemcpy( a_d,a, sizeof(uint32_t)*rows*cols, cudaMemcpyHostToDevice);
cudaMemcpy( b_d,b, sizeof(uint32_t)*rows*cols, cudaMemcpyHostToDevice);
cudaMemcpy( c_d,c, sizeof(uint32_t)*rows*cols, cudaMemcpyHostToDevice);


// указатель измененности
uint32_t *is_changed_host = allocate_matrix_host(1,1);
*is_changed_host = NOT_CHANGED;

cudaMemcpyToSymbol(is_changed_matrix, is_changed_host, sizeof(uint32_t),0,cudaMemcpyHostToDevice);




// настройка ядер для функции создания таблиц
dim3 dimBlock_table_kernel(BLOCK_SIZE,BLOCK_SIZE);// для всех вызовов создания таблиц

//настройка для таблиц из полного цикла
uint32_t grid_x = table_cols_n/BLOCK_SIZE;
if(table_cols_n%BLOCK_SIZE!=0) grid_x++;
uint32_t grid_y = rows/(BLOCK_SIZE*K);
if(rows%(BLOCK_SIZE*K)!=0) grid_y++;
dim3 dimGrid_table_nums(grid_x,grid_y); //для запуска по батчам создание таблиц

// настройка для последней таблицы
grid_x = table_cols_last/BLOCK_SIZE;
if(table_cols_last % BLOCK_SIZE!=0) grid_x++;
dim3 dimGrid_table_last(grid_x,grid_y);

// настройка ядер для умножения
dim3 dimBlock_m4ri(BLOCK_SIZE_COL,BLOCK_SIZE_ROW);

// настройка для умножения из полного цикла
grid_x = table_cols_n /BLOCK_SIZE_COL;
if(table_cols_n%BLOCK_SIZE_COL!=0)grid_x++;
grid_y = rows/BLOCK_SIZE_ROW;
if(rows%BLOCK_SIZE_ROW!=0)grid_y++;
dim3 dimGrid_m4ri_nums(grid_x,grid_y);

// настройка для последнего умножения
grid_x = table_cols_last/BLOCK_SIZE_COL;
if(table_cols_last%BLOCK_SIZE_COL!=0)grid_x++;
dim3 dimGrid_m4ri_last(grid_x,grid_y);

//allocate tables
uint32_t ** tables_n;// = allocate_tables(num_tables,257*table_cols_n);
uint32_t ** tables_last;// = allocate_tables(num_tables,257*table_cols_last);
if(num_launches != 0){
tables_n = allocate_tables(num_tables,256,table_cols_n);

}
if(table_cols_last != 0){
tables_last = allocate_tables(num_tables,256,table_cols_last);
}

for(int i = 0;i < num_launches;i++){
make_table<<<dimGrid_table_nums,dimBlock_table_kernel>>>
(b_d,tables_n,table_cols_n,rows,num_tables,cols,i*table_cols_n);
cudaDeviceSynchronize();
m4ri_mul<<<dimGrid_m4ri_nums,dimBlock_m4ri>>>
(c_d,a_d,tables_n,rows,cols,table_cols_n,cols/BLOCK_SIZE_COL,cols%BLOCK_SIZE_COL,i*table_cols_n);
cudaDeviceSynchronize();
}

if(table_cols_last != 0){

make_table<<<dimGrid_table_last,dimBlock_table_kernel>>>
(b_d,tables_last,table_cols_last,rows,num_tables,cols,num_launches*table_cols_n);
cudaDeviceSynchronize();
m4ri_mul<<<dimGrid_m4ri_last,dimBlock_m4ri>>>
(c_d,a_d,tables_last,rows,cols,table_cols_last,cols/BLOCK_SIZE_COL,cols%BLOCK_SIZE_COL,num_launches*table_cols_n);
cudaDeviceSynchronize();
}

if(num_launches!=0){
delete_tables(tables_n, num_launches);
}
if(table_cols_last!=0){
delete_tables(tables_last,table_cols_last);
}

cudaMemcpy( a,a_d, sizeof(uint32_t)*rows*cols, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();
cudaMemcpyFromSymbol(is_changed_host,is_changed_matrix, sizeof(uint32_t), 0,cudaMemcpyDeviceToHost);
cudaFree(a_d);
cudaFree(b_d);
cudaFree(c_d);
int flag = *is_changed_host;

cudaFreeHost(is_changed_host);
return flag;
}


class Grammar {
public:
    std::set<String> nonterminalSet;
    std::set<String> terminalSet;
    std::map<String, std::set<std::pair<String, String >>> productionsDouble;
    std::map<std::pair<String, String>, std::set<String >> reverseProductionsDouble;
    std::map<String, std::set<String>> productionsUnary;//NonTerminal-> Set of Terminal
    std::map<String, std::set<String>> reverseProductionsUnary;// Terminal -> Set of non terminal
    std::map<String, std::set<String>> nonTerminalToBodyOfProduction;// NonTerminal ->Set nont termianl aka nontermina+ eleme from set or vice versa is key for reverseProduction


    void parse_grammar(const String &filename, const String &delimiter = " ") {
        std::ifstream file(filename);
        if (file.is_open()) {
            std::string line;
            while (getline(file, line)) {
                process_grammar_line(line, delimiter);
            }
            file.close();
        }
        make_reverse_relations();
        make_nonTerminalToBodyOfProduction();
    }


private:
    void make_reverse_relations() {
        //reverseProductionUnary
        make_unary_reverse_relation();
        make_double_reverse_relation();
    }

    void process_grammar_line(String line, const String &delimiter = " ") {
        size_t pos = 0;
        std::string token[2];
        int c = 0;
        while ((pos = line.find(delimiter)) != std::string::npos) {

            token[c] = line.substr(0, pos);
            line.erase(0, pos + delimiter.length());
            c++;
        }
        String head = token[0];

        if (c == 2) {
            String left_terminal = token[1];
            String right_terminal = line;
            auto tail = make_pair(left_terminal, right_terminal);

            this->nonterminalSet.insert(head);// нетерминалы множество
            this->nonterminalSet.insert(left_terminal);
            this->nonterminalSet.insert(right_terminal);
            if (this->productionsDouble.count(head) == 1) { // продукции
                auto iter = this->productionsDouble.find(head);
                iter->second.insert(tail);
            } else {
                this->productionsDouble.insert(make_pair(head, set<pair<String, String >>({tail})));
            }
        } else if (c == 1) {
            const String &terminal = line;
            this->nonterminalSet.insert(head);
            if (this->productionsUnary.count(head) == 1) {
                auto iter = this->productionsUnary.find(head);
                iter->second.insert(terminal);
            } else {
                this->productionsUnary.insert(make_pair(head, set<String>({terminal})));
            }
            this->terminalSet.insert(terminal);
        } else {
            throw "Error while process line from grammar";
        }
    }

    void make_unary_reverse_relation() {
        for (auto nonterminal: this->productionsUnary) {
            for (auto terminal: nonterminal.second) {
                if (reverseProductionsUnary.count(terminal) == 1) {
                    reverseProductionsUnary.find(terminal)->second.insert(nonterminal.first);
                } else {
                    reverseProductionsUnary.insert(make_pair(terminal, set<String>({nonterminal.first})));
                }
            }
        }
    }

    void make_double_reverse_relation() {
        for (auto head:this->productionsDouble) {
            for (auto elem_pair:head.second) {
                if (reverseProductionsDouble.count(elem_pair) == 1) {
                    reverseProductionsDouble.find(elem_pair)->second.insert(head.first);
                } else {
                    reverseProductionsDouble.insert(make_pair(elem_pair, set<String>({head.first})));
                }
            }
        }

    }

    void make_nonTerminalToBodyOfProduction() {
        for (auto leftNonTerminal: nonterminalSet) {
            for (auto rightNonTerminal:nonterminalSet) {
                auto key = make_pair(leftNonTerminal, rightNonTerminal);
                if (reverseProductionsDouble.count(key)) {
                    if (nonTerminalToBodyOfProduction.count(leftNonTerminal)) {
                        nonTerminalToBodyOfProduction.find(leftNonTerminal)->second.insert(rightNonTerminal);
                    } else {
                        nonTerminalToBodyOfProduction.insert(
                                make_pair(leftNonTerminal, set<String>({rightNonTerminal})));
                    }
                    if (nonTerminalToBodyOfProduction.count(rightNonTerminal)) {
                        nonTerminalToBodyOfProduction.find(rightNonTerminal)->second.insert(leftNonTerminal);
                    } else {
                        nonTerminalToBodyOfProduction.insert(
                                make_pair(rightNonTerminal, set<String>({leftNonTerminal})));
                    }
                } else {
                }
            }

        }

    }
};






class Edge {
public:
    int from;
    set<String> label;
    int to;

    Edge(int from, int to) {
        this->from = from;
        this->to = to;
    }


};

class Graph {
public:
    vector<Edge> edges;
    int max_number_of_vertex;
    int multiple_by_32; // is maxnumber if maxnumber % 32=0 or max_number+ (32 -maxnumber % 32)

    void parse_graph(const String &filename, const String &delimiter = " ") {
        std::ifstream file(filename);
        int max_vertex = 0;
        if (file.is_open()) {
            std::string line;

            while (getline(file, line)) {
                size_t pos = 0;
                std::string token[2];
                int c = 0;
                while ((pos = line.find(delimiter)) != std::string::npos) {

                    token[c] = line.substr(0, pos);
                    line.erase(0, pos + delimiter.length());
                    c++;
                }
                if (c == 2) {
                    int l = std::stoi(token[0]);
                    int r = std::stoi(line);
                    max_vertex = std::max(std::max(l, r), max_vertex);
                    Edge edge = Edge(l, r);
                    edge.label.insert(token[1]);
                    edges.push_back(edge);
                } else {
                    throw "Error while process line from graph";
                }


            }
            file.close();
        } else{
            throw "Error File not  found";
        }

        max_vertex+=1;// т.к у нас верщины присутствует от 0 до max_vertex включетельно
        max_number_of_vertex = max_vertex;
        if (max_vertex % SQUEEZE == 0) {
            multiple_by_32 = max_vertex;
        } else {
            int quout = max_vertex % SQUEEZE;
            multiple_by_32 = max_vertex + SQUEEZE - quout;
        }
    }

    void replace_terminals_to_noterminals(Grammar &grammar) {
        for (auto &edge : edges) {
            set<String> tmp;
            for (const String &key:edge.label) {
                if (grammar.reverseProductionsUnary.count(key) == 1) {
                    tmp.insert(grammar.reverseProductionsUnary.find(key)->second.begin(),
                               grammar.reverseProductionsUnary.find(key)->second.end());
                }
            }
            edge.label.clear();
            edge.label.insert(tmp.begin(), tmp.end());
        }
    }

};






void delete_matrix_device(uint32_t * matrix) {
    cudaFree(matrix);
}

void delete_matrix_host(uint32_t * matrix) {
    cudaFreeHost(matrix);
}



struct Matrix {
    uint32_t *matrix_host;
    uint32_t *matrix_device;
    uint32_t *matrix_squeezed_host;
    uint32_t *is_changed_host;
};
struct Table {
    uint32_t *table_n;
    uint32_t *table_last;
};






class Solution {
public:
    Graph graph;
    Grammar grammar;
    map<String, Matrix> nonTerminalToMatrix;
    Table table;

    Solution(const String &filename_grammar, const String &filename_graph, const String &delimiter = " ") {
        // add table size as parameter
        graph.parse_graph(filename_graph, delimiter);
        grammar.parse_grammar(filename_grammar, delimiter);
        graph.replace_terminals_to_noterminals(grammar);
        construct_and_fill_matrices_for_nonterminal_test();

    }
    void output_in(String filename )  {ifstream input(filename);

        vector<String > res(grammar.nonterminalSet.begin(),grammar.nonterminalSet.end());
        sort(res.begin(),res.end());

        ofstream outputfile;
        outputfile.open(filename);
        for (auto &nonterminal: res) {
            auto & matrix = nonTerminalToMatrix.at(nonterminal);
            outputfile << nonterminal;
            bool *bitArray = Decompress(matrix.matrix_host, graph.max_number_of_vertex);
            for (int i = 0; i < graph.max_number_of_vertex; ++i) {
                for (int j = 0; j < graph.max_number_of_vertex; ++j) {
                    if (bitArray[i * graph.max_number_of_vertex + j] != 0) {
                        outputfile << ' ' << i << ' ' << j;
                    }
                }
            }
            outputfile << endl;
        }
        outputfile.close();

    }
    void compute_result() {
        // initial setup
        set<String> changed_matrices = set<String>();
        for (auto &elem: nonTerminalToMatrix) {
            if (*elem.second.is_changed_host == IS_CHANGED) {
                changed_matrices.insert(elem.first);
            }
        }

        if (changed_matrices.empty()) {
            return;//
        }

        while (true) {
            set<String> new_changed_matrices = set<String>();

            for (auto &nonterminal: changed_matrices) {
                if (grammar.nonTerminalToBodyOfProduction.count(nonterminal)) {
                    auto const &possibly_second_key_set = grammar.nonTerminalToBodyOfProduction.find(
                            nonterminal)->second;
                    // перемножаем все пары матриц, в теле которых стоит этот нетерминал  если он там присутствует
                    for (const auto &sec: possibly_second_key_set) {
                        auto key1 = make_pair(nonterminal, sec);
                        auto key2 = make_pair(sec, nonterminal);
                        if (grammar.reverseProductionsDouble.count(key1)) {
                            auto iter = grammar.reverseProductionsDouble.find(key1);
                            for (const auto &res: iter->second) {
                                auto is_changed = perform_matrix_mul(res, iter->first.first, iter->first.second);
                                if (is_changed) {
                                    new_changed_matrices.insert(res);
                                }
                            }
                        }
                        if (grammar.reverseProductionsDouble.count(key2)) {
                            auto iter = grammar.reverseProductionsDouble.find(key2);
                            for (const auto &res: iter->second) {
                                auto is_changed = perform_matrix_mul(res, iter->first.first, iter->first.second);
                                if (is_changed) {
                                    new_changed_matrices.insert(res);
                                }
                            }
                        }
                    }
                }


            }

            if (new_changed_matrices.empty()) {
                //copy
                break;
            }   else {
                changed_matrices = new_changed_matrices;
                //update matrices

            }
            //transfer



        }


    }
    int packedByBlocksNumber(int N, int size) {
        return (N / size + (N % size == 0 ? 0 : 1));
    }

    bool * Decompress(uint32_t * c_arr, uint32_t N) {
        // int num_rows = N;
        int num_columns = packedByBlocksNumber(N, 32);
        bool * arr = reinterpret_cast<bool *>(calloc(N * N, sizeof(bool)));

        uint32_t el;
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                el = c_arr[r * num_columns + (c / 32)];
                if (el & (1 << (31 - (c % 32)))) {
                    arr[r * N + c] = 1;
                }
            }
        }

        return arr;
    }



private:

    void construct_and_fill_matrices_for_nonterminal_test() {
        int rows = this->graph.multiple_by_32;
        int cols = this->graph.multiple_by_32/32;
        for (auto nonterminal: grammar.nonterminalSet) {
            Matrix matrix = Matrix();
            matrix.matrix_host = allocate_matrix_host(rows,cols); //alloc_matrix_host_with_zeros(rows, cols);
            matrix.is_changed_host = allocate_matrix_host(1,1);
            *matrix.is_changed_host = NOT_CHANGED;
            this->nonTerminalToMatrix.insert(make_pair(nonterminal, matrix));
        }// заполнили нулями для хоста
        for (auto &edge:graph.edges) {
            auto i = edge.from;
            auto j = edge.to;
            for (const auto &nonterminal:edge.label) { // заполнилии 1 в i,j для матриц на метках из i в j есть этот нетерминал
                auto &matrix = this->nonTerminalToMatrix.find(nonterminal)->second;
                write_bit(matrix.matrix_host,i,j,cols);
                if (*matrix.is_changed_host == NOT_CHANGED) {
                    *matrix.is_changed_host = IS_CHANGED;
                }
            }
        }
    }


    void write_bit(uint32_t *m, int i, int j,int cols){
        m[i * cols + (j / 32)] |= (1 << (31 - (j % 32)));
    }



    // A = C*B
    int perform_matrix_mul(const String &head, const String &left, const String &right) {
        int rows = graph.multiple_by_32;
        int cols = graph.multiple_by_32/32;
        auto &A = this->nonTerminalToMatrix.at(head);
        auto &C = this->nonTerminalToMatrix.at(left);
        auto &B = this->nonTerminalToMatrix.at(right);
        *A.is_changed_host = 0;
        // a =cb и таков порядок аргубемнов
        int res = wrapper_m4ri(A.matrix_host,C.matrix_host,B.matrix_host,rows,cols);
        *A.is_changed_host =  res;
        return res;
    }

};


int main(int argc, char* argv[]) {
    auto solution = Solution(argv[1], argv[2], DELIMITR);
    clock_t begin = clock();
    solution.compute_result();
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    solution.output_in(argv[3]);
    cout<<elapsed_secs<<endl;


}
