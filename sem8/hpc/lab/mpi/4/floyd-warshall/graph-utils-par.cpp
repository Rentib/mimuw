/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#include <iostream>
#include <cassert>
#include <mpi.h>
#include "graph-base.h"
#include "graph-utils.h"

int getFirstGraphRowOfProcess(int numVertices, int numProcesses, int myRank) {
    int blockSize = numVertices / numProcesses;
    int remainder = numVertices % numProcesses;

    if (myRank <= remainder) {
        blockSize += 1;
        return myRank * blockSize;
    } else {
        return myRank * blockSize + remainder;
    }
}

Graph* createAndDistributeGraph(int numVertices, int numProcesses, int myRank) {
    assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);

    auto graph = allocateGraphPart(
            numVertices,
            getFirstGraphRowOfProcess(numVertices, numProcesses, myRank),
            getFirstGraphRowOfProcess(numVertices, numProcesses, myRank + 1)
    );

    if (graph == nullptr) {
        return nullptr;
    }

    assert(graph->numVertices > 0 && graph->numVertices == numVertices);
    assert(graph->firstRowIdxIncl >= 0 && graph->lastRowIdxExcl <= graph->numVertices);

    if (myRank == 0) {
        for (int i = graph->firstRowIdxIncl; i < graph->lastRowIdxExcl; ++i) {
            initializeGraphRow(graph->data[i], i, graph->numVertices);
        }

        for (int i = graph->lastRowIdxExcl, rcvRank = 1; i < graph->numVertices; ++i) {
            int nextRankFirstRow = getFirstGraphRowOfProcess(numVertices, numProcesses, rcvRank + 1);

            if (i == nextRankFirstRow) {
                rcvRank++;
            }

            initializeGraphRow(graph->extraRow, i, graph->numVertices);
//            std::cout << "Sending row " << i << " to receiver " << rcvRank << std::endl;
            MPI_Send(graph->extraRow, graph->numVertices, MPI_INT, rcvRank, 0, MPI_COMM_WORLD);
        }
    } else {
        for (int i = 0, r = graph->firstRowIdxIncl; r < graph->lastRowIdxExcl; ++i, ++r) {
//            std::cout << "Getting row " << r << std::endl;
            MPI_Recv(graph->data[i], graph->numVertices, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    return graph;
}

void collectAndPrintGraph(Graph* graph, int numProcesses, int myRank) {
    assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);
    assert(graph->numVertices > 0);
    assert(graph->firstRowIdxIncl >= 0 && graph->lastRowIdxExcl <= graph->numVertices);

    if (myRank == 0) {
        for (int i = graph->firstRowIdxIncl; i < graph->lastRowIdxExcl; ++i) {
            printGraphRow(graph->data[i], i, graph->numVertices);
        }

        for (int i = graph->lastRowIdxExcl, rcvRank = 1; i < graph->numVertices; ++i) {
            int nextRankFirstRow = getFirstGraphRowOfProcess(graph->numVertices, numProcesses, rcvRank + 1);

            if (i == nextRankFirstRow) {
                rcvRank++;
            }

            MPI_Recv(graph->extraRow, graph->numVertices, MPI_INT, rcvRank, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printGraphRow(graph->extraRow, i, graph->numVertices);
        }
    } else {
        for (int i = 0, r = graph->firstRowIdxIncl; r < graph->lastRowIdxExcl; ++i, ++r) {
            MPI_Send(graph->data[i], graph->numVertices, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
}

void destroyGraph(Graph* graph, int numProcesses, int myRank) {
    freeGraphPart(graph);
}
