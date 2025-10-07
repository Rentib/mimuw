/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#include "graph-base.h"
#include "graph-utils.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <mpi.h>

int getFirstGraphRowOfProcess(int numVertices, int numProcesses, int myRank)
{
    int base = numVertices / numProcesses;
    int remainder = numVertices % numProcesses;
    return myRank * base + std::min(myRank, remainder);
}

Graph *createAndDistributeGraph(int numVertices, int numProcesses, int myRank)
{
    assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);

    auto graph = allocateGraphPart(
        numVertices,
        getFirstGraphRowOfProcess(numVertices, numProcesses, myRank),
        getFirstGraphRowOfProcess(numVertices, numProcesses, myRank + 1));

    if (graph == nullptr) {
        return nullptr;
    }

    assert(graph->numVertices > 0 && graph->numVertices == numVertices);
    assert(graph->firstRowIdxIncl >= 0 &&
           graph->lastRowIdxExcl <= graph->numVertices);

    if (myRank == 0) {
        for (int destRank = 0; destRank < numProcesses; ++destRank) {
            int destFirst =
                getFirstGraphRowOfProcess(numVertices, numProcesses, destRank);
            int destLast = getFirstGraphRowOfProcess(numVertices, numProcesses,
                                                     destRank + 1);
            int numRows = destLast - destFirst;

            Graph *foreignGraph =
                allocateGraphPart(numVertices, destFirst, destLast);

            for (int i = 0; i < numRows; ++i) {
                initializeGraphRow(foreignGraph->data[i], destFirst + i,
                                   numVertices);
            }

            if (destRank == 0) {
                for (int i = 0; i < numRows; ++i) {
                    memcpy(graph->data[i], foreignGraph->data[i],
                           numVertices * sizeof(int));
                }
            } else {
                for (int i = 0; i < numRows; ++i) {
                    MPI_Send(foreignGraph->data[i], numVertices, MPI_INT,
                             destRank, 0, MPI_COMM_WORLD);
                }
            }

            freeGraphPart(foreignGraph);
        }

    } else {
        int numRows = graph->lastRowIdxExcl - graph->firstRowIdxIncl;
        for (int i = 0; i < numRows; ++i) {
            MPI_Recv(graph->data[i], numVertices, MPI_INT, 0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
    }

    return graph;
}

void collectAndPrintGraph(Graph *graph, int numProcesses, int myRank)
{
    assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);
    assert(graph->numVertices > 0);
    assert(graph->firstRowIdxIncl >= 0 &&
           graph->lastRowIdxExcl <= graph->numVertices);

    if (myRank == 0) {
        int **fullGraph = new int *[graph->numVertices];
        for (int i = 0; i < graph->numVertices; ++i) {
            fullGraph[i] = new int[graph->numVertices];
        }

        int numOwnRows = graph->lastRowIdxExcl - graph->firstRowIdxIncl;
        for (int i = 0; i < numOwnRows; ++i) {
            memcpy(fullGraph[graph->firstRowIdxIncl + i], graph->data[i],
                   graph->numVertices * sizeof(int));
        }

        for (int srcRank = 1; srcRank < numProcesses; ++srcRank) {
            int srcFirst = getFirstGraphRowOfProcess(graph->numVertices,
                                                     numProcesses, srcRank);
            int srcLast = getFirstGraphRowOfProcess(graph->numVertices,
                                                    numProcesses, srcRank + 1);
            int numSrcRows = srcLast - srcFirst;

            for (int i = 0; i < numSrcRows; ++i) {
                MPI_Recv(fullGraph[srcFirst + i], graph->numVertices, MPI_INT,
                         srcRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        for (int i = 0; i < graph->numVertices; ++i) {
            printGraphRow(fullGraph[i], i, graph->numVertices);
        }

        for (int i = 0; i < graph->numVertices; ++i) {
            delete[] fullGraph[i];
        }
        delete[] fullGraph;
    } else {
        int numRows = graph->lastRowIdxExcl - graph->firstRowIdxIncl;
        for (int i = 0; i < numRows; ++i) {
            MPI_Send(graph->data[i], graph->numVertices, MPI_INT, 0, 0,
                     MPI_COMM_WORLD);
        }
    }
}

void destroyGraph(Graph *graph, int numProcesses, int myRank)
{
    freeGraphPart(graph);
}
