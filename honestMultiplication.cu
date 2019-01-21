
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




uint32_t * allocate_matrix_host(int rows,int cols) {
    // allocate memory in host RAM
    uint32_t *matrix;
    cudaMallocHost((void **) &matrix, sizeof(uint32_t)*rows * cols);
    return matrix;
}

uint32_t * allocate_matrix_device(int rows,int cols){
    uint32_t *matrix;
    cudaMalloc((void **) &matrix, sizeof(uint32_t)*rows*cols);
    return matrix;

}


void delete_matrix_device(uint32_t * matrix) {
    cudaFree(matrix);
}

void delete_matrix_host(uint32_t * matrix) {
    cudaFreeHost(matrix);
}



//__device__ is_changed = 0;
__global__ void gpu_matrix_mult(uint32_t *a,uint32_t *b, uint32_t *c, int m, int n, int k,uint32_t * is_changed)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t sum = 0;
    uint32_t old_c;
    
    if( col < k && row < m)
    {
        old_c = c[row*k+col];
        sum = 0;
        for(int i = 0; i < n; i++)
        {
            sum |= a[row * n + i] & b[i * k + col];
        }
        sum|=old_c;
        if(*is_changed == NOT_CHANGED && sum!=old_c ) {
            
            *is_changed = IS_CHANGED;
        }
        c[row * k + col] = sum;
    }

//    uint32_t value = 0;
    //
    //                for (int k = 0; k < row_b; k++) {
    //                    value |= a[i * row_b + k] & b[k * col_b + j];
    //                }
    //                if (*is_changed == NOT_CHANGED && (c[i * col_b + j] | value) != c[i * col_b + j]) {
    //                    *is_changed = IS_CHANGED;
    //                }
    //                c[i * col_b + j] |=
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
    uint32_t * extra_matrix;
    Table table;

    Solution(const String &filename_grammar, const String &filename_graph, const String &delimiter = " ") {
        // add table size as parameter
        graph.parse_graph(filename_graph, delimiter);
        grammar.parse_grammar(filename_grammar, delimiter);
        graph.replace_terminals_to_noterminals(grammar);
        construct_and_fill_matrices_for_nonterminal_test();

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

private:


    // не забудь здесь выставить флаги для тех матриц, в которых не нули
    // void construct_and_fill_matrices_for_nonterminals() {
    //     int rows = this->graph.multiple_by_32;
    //     int cols = this->graph.multiple_by_32 / SQUEEZE;  // сжимаем по строкам
    //     for (auto nonterminal: grammar.nonterminalSet) {
    //         Matrix matrix = Matrix();
    //         matrix.matrix_host = alloc_matrix_host_with_zeros(rows, cols);
    //         matrix.is_changed_host = alloc_matrix_host_with_zeros(1, 1);
    //         this->nonTerminalToMatrix.insert(make_pair(nonterminal, matrix));
    //         matrix.matrix_device = alloc_matrix_device_with_zeros(rows, cols);// на гпу
    //     }// заполнили нулями для хоста

    //     for (auto &edge:graph.edges) {
    //         auto i = edge.from;
    //         auto j = edge.to;
    //         for (const auto &nonterminal:edge.label) { // заполнилии 1 в i,j для матриц на метках из i в j есть этот нетерминал
    //             fill_squeezed_matrix(this->nonTerminalToMatrix.find(nonterminal)->second.matrix_host, i, j,
    //                                  graph.multiple_by_32);
    //         }
    //     }

    //     for (const auto &nonterminal: grammar.nonterminalSet) {//трансфер данные с цпу на гпу
    //         auto &matrix = this->nonTerminalToMatrix.find(nonterminal)->second;
    //         transfer_matrix_from_host_to_gpu(matrix.matrix_host, matrix.matrix_device, rows, cols);
    //     }
    // }

    void construct_and_fill_matrices_for_nonterminal_test() {
        int rows = this->graph.max_number_of_vertex;
        int cols = this->graph.max_number_of_vertex;
        int squeezed_cols = this->graph.multiple_by_32;
        for (auto nonterminal: grammar.nonterminalSet) {
            Matrix matrix = Matrix();
            matrix.matrix_host = allocate_matrix_host(rows,cols); //alloc_matrix_host_with_zeros(rows, cols);
           // matrix.matrix_squeezed_host = new uint32_t[rows*squeezed_cols];
            matrix.is_changed_host = allocate_matrix_host(1,1);
            *matrix.is_changed_host = NOT_CHANGED;

            this->nonTerminalToMatrix.insert(make_pair(nonterminal, matrix));
        }// заполнили нулями для хоста
        extra_matrix = allocate_matrix_host(cols,rows); // аллок памяти для доп матрицы
        for (auto &edge:graph.edges) {
            auto i = edge.from;
            auto j = edge.to;
            for (const auto &nonterminal:edge.label) { // заполнилии 1 в i,j для матриц на метках из i в j есть этот нетерминал
                auto &matrix = this->nonTerminalToMatrix.find(nonterminal)->second;
                matrix.matrix_host[i * cols + j] = 1;
                //write_bit(matrix.matrix_squeezed_host,i,j,squeezed_cols);
                if (*matrix.is_changed_host == NOT_CHANGED) {
                    *matrix.is_changed_host = IS_CHANGED;
                }
            }
        }
    }


    void write_bit(uint32_t *m, int i, int j,int cols){
//        m[i * cols + (j / 32)] |= (1ULL << (31 - (j % 32)));
        m[i * cols + (j / 32)] |= (1 << (31 - (j % 32)));
    }

    inline void fill_squeezed_matrix(uint32_t *matrix, int i, int j, int size32) {
        // строка ок
        int cols = size32 / 32;
        int position_in_number32 = (SQUEEZE - 1) - (j % SQUEEZE);
        int position_in_squezzed_row = j / 32;
        matrix[i * cols + position_in_squezzed_row] |= (1L << position_in_number32);
    }

    // uint32_t *alloc_matrix_host_with_zeros(int rows, int cols) {
    // }

    // uint32_t *alloc_matrix_device_with_zeros(int rows, int cols) {
    // }

    void transfer_matrix_from_host_to_gpu(uint32_t *host, uint32_t *device, int rows, int cols) {
        //
    }

    void transfer_matrix_from_gpu_to_host(uint32_t *device, uint32_t *host, int rows, int cols) {

    }


    void gpu_version(const uint32_t *a, const uint32_t *b, uint32_t *c, int n, uint32_t *is_changed){

        // c += ab
      //  cout<<"H";

        uint32_t * a_d  = allocate_matrix_device(n,n);
        uint32_t * b_d  = allocate_matrix_device(n,n);
        uint32_t * c_d  = allocate_matrix_device(n,n);
        uint32_t * flag_device = allocate_matrix_device(1,1);

        cudaMemcpy( a_d,a, sizeof(uint32_t)*n*n, cudaMemcpyHostToDevice);
        cudaMemcpy( b_d,b, sizeof(uint32_t)*n*n, cudaMemcpyHostToDevice);
        cudaMemcpy( c_d,c, sizeof(uint32_t)*n*n, cudaMemcpyHostToDevice);
        cudaMemcpy( flag_device,is_changed, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();


        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        gpu_matrix_mult<<<dimGrid,dimBlock>>>(a_d,b_d, c_d, n,  n, n,flag_device);
        cudaDeviceSynchronize();


        cudaMemcpy( c,c_d, sizeof(uint32_t)*n*n, cudaMemcpyDeviceToHost);
        cudaMemcpy( is_changed,flag_device, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        delete_matrix_device(a_d);
        delete_matrix_device(b_d);
        delete_matrix_device(c_d);
        delete_matrix_device(flag_device);

    }

    // c = ab
    void dummy_subring_matrix_mul(const uint32_t *a, int row_a, int col_a, const uint32_t *b, int row_b, int col_b,
                                  uint32_t *c, uint32_t *is_changed) {
        if (col_a != row_b) {
            printf("The matrices can't be multiplied with each other.\n");
            return;
        }
        gpu_version(a,b,c,row_a,is_changed);
//
//        for (int i = 0; i < row_a; i++) {
//
//            for (int j = 0; j < col_b; j++) {
//                uint32_t value = 0;
//
//                for (int k = 0; k < row_b; k++) {
//                    value |= a[i * row_b + k] & b[k * col_b + j];
//                }
//                if (*is_changed == NOT_CHANGED && (c[i * col_b + j] | value) != c[i * col_b + j]) {
//                    *is_changed = IS_CHANGED;
//                }
//                c[i * col_b + j] |= value;
//            }
//        }
    }
    // perform algo

    //
    // allocate matrices and tables on device

    //

    // A = C*B
    int perform_matrix_mul(const String &head, const String &left, const String &right) {
        int rows = graph.max_number_of_vertex;
        int cols = graph.max_number_of_vertex;
        auto &A = this->nonTerminalToMatrix.at(head);
        auto &C = this->nonTerminalToMatrix.at(left);
        auto &B = this->nonTerminalToMatrix.at(right);
        *A.is_changed_host = 0;
        if (head == left) {// нужно создать доп матрицу т.к A = C
            copy(C.matrix_host, C.matrix_host + rows * cols, extra_matrix);
            dummy_subring_matrix_mul(extra_matrix, rows, cols, B.matrix_host, rows, cols, A.matrix_host,
                                     A.is_changed_host);
        }
        if (head == right) {//нужно создать доп матрицу т.к A = B
            copy(B.matrix_host, B.matrix_host + rows * cols, extra_matrix);
            dummy_subring_matrix_mul(C.matrix_host, rows, cols, extra_matrix, rows, cols, A.matrix_host,
                                     A.is_changed_host);
        } else {
            dummy_subring_matrix_mul(C.matrix_host, rows, cols, B.matrix_host, rows, cols, A.matrix_host,
                                     A.is_changed_host);


        }
        return *A.is_changed_host;
    }

};


int main() {
    auto solution = Solution("GPPerf1_cnf.txt", "wine_right.rdf.txt", DELIMITR);
    clock_t begin = clock();
    solution.compute_result();
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    vector<String > res(solution.grammar.nonterminalSet.begin(),solution.grammar.nonterminalSet.end());
    sort(res.begin(),res.end());
    int cols = solution.graph.max_number_of_vertex;
    for(auto nonterminal: res){
        cout<<nonterminal;
        auto & matrix = solution.nonTerminalToMatrix.at(nonterminal);
        for(int i=0;i<solution.graph.max_number_of_vertex;i++){
            for(int j =0;j<solution.graph.max_number_of_vertex;j++) {
                if(matrix.matrix_host[i*cols + j]!=0) {
                    cout<<" "<<i<<" "<<j;
                }
            }
        }
        cout<<endl;
     }
//    cout<<elapsed_secs<<endl;


}
