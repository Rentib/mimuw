/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#include "graph-base.h"
#include "graph-utils.h"
#include <cassert>
#include <iostream>
#include <mpi.h>
#include <string>

static void runFloydWarshallParallel(Graph *graph, int numProcesses, int myRank)
{
    assert(numProcesses <= graph->numVertices);

    int m = graph->numVertices;
    int firstRow = graph->firstRowIdxIncl;
    int lastRow = graph->lastRowIdxExcl;

    int base = m / numProcesses;
    int remainder = m % numProcesses;

    for (int k = 0; k < m; ++k) {
        int k_rank;
        if (k < remainder * (base + 1)) {
            k_rank = k / (base + 1);
        } else {
            k_rank = remainder + (k - remainder * (base + 1)) / base;
        }

        int *row_k;
        if (myRank == k_rank) {
            int ownerFirstRow =
                getFirstGraphRowOfProcess(m, numProcesses, k_rank);
            int local_k = k - ownerFirstRow;
            row_k = graph->data[local_k];
            MPI_Bcast(row_k, m, MPI_INT, k_rank, MPI_COMM_WORLD);
        } else {
            MPI_Bcast(graph->extraRow, m, MPI_INT, k_rank, MPI_COMM_WORLD);
            row_k = graph->extraRow;
        }

        for (int i_local = 0; i_local < (lastRow - firstRow); ++i_local) {
            int *row_i = graph->data[i_local];
            for (int j = 0; j < m; ++j) {
                int sum = row_i[k] + row_k[j];
                if (row_i[j] > sum) {
                    row_i[j] = sum;
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int numVertices = 0;
    int numProcesses = 0;
    int myRank = 0;
    int showResults = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

#ifdef USE_RANDOM_GRAPH
#ifdef USE_RANDOM_SEED
    srand(USE_RANDOM_SEED);
#endif
#endif

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]).compare("--show-results") == 0) {
            showResults = 1;
        } else {
            numVertices = std::stoi(argv[i]);
        }
    }

    if (numVertices <= 0) {
        std::cerr << "Usage: " << argv[0] << "  [--show-results] <num_vertices>"
                  << std::endl;
        MPI_Finalize();
        return 1;
    }

    if (numProcesses > numVertices) {
        numProcesses = numVertices;

        if (myRank >= numProcesses) {
            MPI_Finalize();
            return 0;
        }
    }

    std::cerr << "Running the Floyd-Warshall algorithm for a graph with "
              << numVertices << " vertices." << std::endl;

    auto graph = createAndDistributeGraph(numVertices, numProcesses, myRank);
    if (graph == nullptr) {
        std::cerr << "Error distributing the graph for the algorithm."
                  << std::endl;
        MPI_Finalize();
        return 2;
    }

    if (showResults) {
        collectAndPrintGraph(graph, numProcesses, myRank);
    }

    double startTime = MPI_Wtime();

    runFloydWarshallParallel(graph, numProcesses, myRank);

    double endTime = MPI_Wtime();

    std::cerr << "The time required for the Floyd-Warshall algorithm on a "
              << numVertices << "-node graph with " << numProcesses
              << " process(es): " << endTime - startTime << std::endl;

    if (showResults) {
        collectAndPrintGraph(graph, numProcesses, myRank);
    }

    destroyGraph(graph, numProcesses, myRank);

    MPI_Finalize();

    return 0;
}
