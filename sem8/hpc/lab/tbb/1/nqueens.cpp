// Solves the n-quees puzzle on an n x x checkerboard.
//
// This sequential implementation is to be extended with TBB to get a
// parallel implementation.
//
// HPC course, MIM UW
// Krzysztof Rzadca, LGPL

#include "tbb/tbb.h"
#include <cmath>
#include <iostream>
#include <list>
#include <vector>

// Indexed by column. Value is the row where the queen is placed,
// or -1 if no queen.
typedef std::vector<int> Board;

void pretty_print(const Board &board)
{
    for (int row = 0; row < (int)board.size(); row++) {
        for (const auto &loc : board) {
            if (loc == row)
                std::cout << "*";
            else
                std::cout << " ";
        }
        std::cout << std::endl;
    }
}

// Checks the location of queen in column 'col' against queens in cols [0, col).
bool check_col(Board &board, int col_prop)
{
    int row_prop = board[col_prop];
    int col_queen = 0;
    for (auto i = board.begin(); (i != board.end()) && (col_queen < col_prop);
         ++i, ++col_queen) {
        int row_queen = *i;
        if (row_queen == row_prop) {
            return false;
        }
        if (abs(row_prop - row_queen) == col_prop - col_queen) {
            return false;
        }
    }
    return true;
}

void initialize(Board &board, int size)
{
    board.reserve(size);
    for (int col = 0; col < size; ++col)
        board.push_back(-1);
}

// Solves starting from a partially-filled board (up to column col).
void recursive_solve(Board &partial_board, int col, std::list<Board> &solutions)
{
    int b_size = partial_board.size();
    if (col == b_size) {
        solutions.push_back(partial_board);
    } else {
        for (int tested_row = 0; tested_row < b_size; tested_row++) {
            partial_board[col] = tested_row;
            if (check_col(partial_board, col))
                recursive_solve(partial_board, col + 1, solutions);
        }
    }
}

template <int D>
void recursive_solve_parallel(Board &partial_board, int col,
                              tbb::concurrent_queue<Board> &solutions)
{
    int b_size = partial_board.size();
    if (col == b_size) {
        solutions.push(partial_board);
    } else if (col < D) {
        // Parallelize the first few levels of the search tree
        tbb::parallel_for(0, b_size, [&](int tested_row) {
            Board local_board = partial_board; // Make a local copy
            local_board[col] = tested_row;
            if (check_col(local_board, col)) {
                recursive_solve_parallel<D>(local_board, col + 1, solutions);
            }
        });
    } else {
        // Switch to sequential for deeper levels to avoid too much overhead
        for (int tested_row = 0; tested_row < b_size; tested_row++) {
            partial_board[col] = tested_row;
            if (check_col(partial_board, col)) {
                recursive_solve_parallel<D>(partial_board, col + 1, solutions);
            }
        }
    }
}

void run_sequential(int board_size)
{
    Board board{};
    initialize(board, board_size);
    std::list<Board> solutions{};
    tbb::tick_count seq_start_time = tbb::tick_count::now();
    recursive_solve(board, 0, solutions);
    tbb::tick_count seq_end_time = tbb::tick_count::now();
    double seq_time = (seq_end_time - seq_start_time).seconds();
    std::cout << "seq time: " << seq_time << "[s]" << std::endl;
    std::cout << "solution count: " << solutions.size() << std::endl;
}

template <int D> void run_parallel(int board_size)
{
    Board board{};
    initialize(board, board_size);
    tbb::concurrent_queue<Board> solutions{};
    tbb::tick_count par_start_time = tbb::tick_count::now();
    recursive_solve_parallel<D>(board, 0, solutions);
    tbb::tick_count par_end_time = tbb::tick_count::now();
    double par_time = (par_end_time - par_start_time).seconds();
    std::cout << "par time[" << D << "]: " << par_time << "[s]" << std::endl;
    std::cout << "solution count: " << solutions.unsafe_size() << std::endl;
}

int main()
{
    const int board_size = 13;

    run_sequential(board_size);
    run_parallel<0>(board_size);
    run_parallel<1>(board_size);
    run_parallel<2>(board_size);
    run_parallel<3>(board_size);
    run_parallel<4>(board_size);
    run_parallel<5>(board_size);
    run_parallel<6>(board_size);
    run_parallel<7>(board_size);
    run_parallel<8>(board_size);
    run_parallel<9>(board_size);

    // for (const auto& sol : solutions) {
    //     pretty_print(sol);
    //     std::cout << std::endl;
    // }
}
