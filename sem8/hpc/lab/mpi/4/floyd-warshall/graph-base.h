/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#ifndef __MY_GRAPH_BASE__H__
#define __MY_GRAPH_BASE__H__

/**
 * A fragment of a graph represented as a matrix.
 */
class Graph {
public:
    int** data;
    int* extraRow;
    int numVertices;
    int firstRowIdxIncl;
    int lastRowIdxExcl;
};

/**
 * Allocates a fragment of a graph matrix.
 * The matrix, as a whole, has numVertices
 * rows and colums. The fragment comprises
 * entire rows from firstRowIdxIncl (inclusive)
 * to lastRowIdxExcl (exclusive).
 */
Graph* allocateGraphPart(int numVertices, int firstRowIdxIncl, int lastRowIdxExcl);

/**
 * Initializes a single row of a graph either with
 * random or deterministically selected elements.
 */
void initializeGraphRow(int* row, int rowIdx, int numVertices);

/**
 * Prints a single row of a graph matrix
 * to the standard output.
 */
void printGraphRow(int const* row, int rowIdx, int numVertices);

/**
 * Frees a fragment of a graph matrix.
 */
void freeGraphPart(Graph* graph);

#endif /* __MY_GRAPH_BASE__H__ */
