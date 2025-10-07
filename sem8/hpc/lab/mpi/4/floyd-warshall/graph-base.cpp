/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "graph-base.h"

Graph* allocateGraphPart(int numVertices, int firstRowIdxIncl, int lastRowIdxExcl) {
    if (firstRowIdxIncl >= lastRowIdxExcl || firstRowIdxIncl < 0 || numVertices <= 0) {
        return nullptr;
    }

    auto graph = new Graph;
    graph->data = nullptr;
    graph->extraRow = nullptr;
    graph->numVertices = numVertices;
    graph->firstRowIdxIncl = firstRowIdxIncl;
    graph->lastRowIdxExcl = lastRowIdxExcl;
    graph->data = new int *[graph->lastRowIdxExcl - graph->firstRowIdxIncl];

    if (graph->data == nullptr) {
        freeGraphPart(graph);
        return nullptr;
    }

    graph->extraRow = new int[graph->numVertices];

    if (graph->extraRow == nullptr) {
        freeGraphPart(graph);
        return nullptr;
    }

    int n = graph->lastRowIdxExcl - graph->firstRowIdxIncl;

    for (int i = 0; i < n; ++i) {
        graph->data[i] = nullptr;
    }

    for (int i = 0; i < n; ++i) {
        graph->data[i] = new int[graph->numVertices];

        if (graph->data[i] == nullptr) {
            freeGraphPart(graph);
            return nullptr;
        }
    }

    return graph;
}

void initializeGraphRow(int* row, int rowIdx, int numVertices) {
    for (int j = 0; j < numVertices; ++j) {
        row[j] = rowIdx == j ? 0 :
#ifndef USE_RANDOM_GRAPH
        ((rowIdx - j == 1 || j - rowIdx == 1) ? 1 : numVertices + 5);
#else
        (rand() & 8191) + 1;
#endif
    }
}

void printGraphRow(int const* row, int rowIdx, int numVertices) {
    std::cout << row[0];

    for (int j = 1; j < numVertices; ++j) {
        std::cout << " " << row[j];
    }

    std::cout << std::endl;
}

void freeGraphPart(Graph* graph) {
    if (graph == nullptr) {
        return;
    }

    if (graph->extraRow != nullptr) {
        delete (graph->extraRow);
        graph->extraRow = nullptr;
    }

    if (graph->data != nullptr) {
        for (int i = 0, n = graph->lastRowIdxExcl - graph->firstRowIdxIncl; i < n; ++i) {
            if (graph->data[i] != nullptr) {
                delete (graph->data[i]);
                graph->data[i] = nullptr;
            }
        }

        delete (graph->data);
        graph->data = nullptr;
    }

    graph->numVertices = 0;
    graph->firstRowIdxIncl = 0;
    graph->lastRowIdxExcl = 0;
}
