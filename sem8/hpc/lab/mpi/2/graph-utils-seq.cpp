/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#include "graph-base.h"
#include "graph-utils.h"
#include <cassert>

Graph *createAndDistributeGraph(int numVertices, int numProcesses, int myRank)
{
	assert(numProcesses == 1 && myRank == 0);

	auto graph = allocateGraphPart(numVertices, 0, numVertices);

	if (graph == nullptr) {
		return nullptr;
	}

	assert(graph->numVertices > 0 && graph->numVertices == numVertices);
	assert(graph->firstRowIdxIncl == 0 &&
	       graph->lastRowIdxExcl == graph->numVertices);

	for (int i = 0; i < graph->numVertices; ++i) {
		initializeGraphRow(graph->data[i], i, graph->numVertices);
	}

	return graph;
}

void collectAndPrintGraph(Graph *graph, int numProcesses, int myRank)
{
	assert(numProcesses == 1 && myRank == 0);
	assert(graph->numVertices > 0);
	assert(graph->firstRowIdxIncl == 0 &&
	       graph->lastRowIdxExcl == graph->numVertices);

	for (int i = 0; i < graph->numVertices; ++i) {
		printGraphRow(graph->data[i], i, graph->numVertices);
	}
}

void destroyGraph(Graph *graph, int numProcesses, int myRank)
{
	freeGraphPart(graph);
}
