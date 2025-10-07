#include "tbb/tbb.h"
#include <cmath>
#include <iostream>
#include <list>
#include <numeric>
#include <vector>

typedef std::vector<int> Board;

void pretty_print(const Board &board)
{
    for (int row = 0; row < (int)board.size(); row++) {
        for (const auto &loc : board) {
            std::cout << (loc == row ? "*" : " ");
        }
        std::cout << std::endl;
    }
}

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

template <int D>
void recursive_solve_parallel(
    Board &partial_board, int col,
    tbb::enumerable_thread_specific<std::list<Board>> &solutions)
{
    int b_size = partial_board.size();
    if (col == b_size) {
        solutions.local().push_back(partial_board);
    } else if (col < D) {
        // Parallelize the first few levels of the search tree
        // tbb::parallel_for(0, b_size, [&](int tested_row) {
        //     Board local_board = partial_board; // Make a local copy
        //     local_board[col] = tested_row;
        //     if (check_col(local_board, col)) {
        //         recursive_solve_parallel<D>(local_board, col + 1, solutions);
        //     }
        // });

        std::vector<int> rows(b_size);
        std::iota(rows.begin(), rows.end(), 0);
        tbb::parallel_for_each(rows.begin(), rows.end(), [&](int tested_row) {
            Board local_board = partial_board;
            local_board[col] = tested_row;
            if (check_col(local_board, col)) {
                recursive_solve_parallel<D>(local_board, col + 1, solutions);
            }
        });
    } else {
        for (int tested_row = 0; tested_row < b_size; tested_row++) {
            partial_board[col] = tested_row;
            if (check_col(partial_board, col)) {
                recursive_solve_parallel<D>(partial_board, col + 1, solutions);
            }
        }
    }
}

template <int D> void run_parallel(int board_size)
{
    Board board{};
    initialize(board, board_size);
    tbb::enumerable_thread_specific<std::list<Board>> solutions;

    tbb::tick_count par_start_time = tbb::tick_count::now();
    recursive_solve_parallel<D>(board, 0, solutions);
    tbb::tick_count par_end_time = tbb::tick_count::now();
    double par_time = (par_end_time - par_start_time).seconds();

    std::list<Board> all_solutions;
    for (auto &local_list : solutions) {
        all_solutions.splice(all_solutions.end(), local_list);
    }

    std::cout << "par time[" << D << "]: " << par_time << "[s]" << std::endl;
    std::cout << "solution count: " << all_solutions.size() << std::endl;

    // Uncomment to print solutions
    // for (const auto &sol : all_solutions) {
    //     pretty_print(sol);
    //     std::cout << std::endl;
    // }
}

int main()
{
    const int board_size = 13;
    run_parallel<7>(board_size);
    return 0;
}
