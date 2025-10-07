/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#ifndef __MY_GRAPH_UTILS__H__
#define __MY_GRAPH_UTILS__H__

#include "graph-base.h"

/**
 * Creates a graph with a given number of
 * vertices distributing it among a given
 * number of processes where the present
 * process has a given rank.
 */
Graph* createAndDistributeGraph(int numVertices, int numProcesses, int myRank);

/**
 * Collects parts of a graph from a
 * number of processes and prints
 * the entire graph to standard output.
 * The rank of the present process is given.
 */
void collectAndPrintGraph(Graph* graph, int numProcesses, int myRank);

/**
 * Destroys parts of a graph at all
 * processes where the rank of the
 * present process is given.
 */
void destroyGraph(Graph* graph, int numProcesses, int myRank);

/**
 * returns the first row (global index) of a process having a given rank
 */
int getFirstGraphRowOfProcess(int numVertices, int numProcesses, int myRank);

#endif /* __MY_GRAPH_UTILS__H__ */
