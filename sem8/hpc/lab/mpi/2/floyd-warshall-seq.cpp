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

static void runFloydWarshallSequential(Graph *graph)
{
	/* For the sequential version, we assume the entire graph. */
	assert(graph->firstRowIdxIncl == 0 &&
	       graph->lastRowIdxExcl == graph->numVertices);
	int m = graph->numVertices;

	for (int k = 0; k < m; ++k) {
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				int pathSum =
				    graph->data[i][k] + graph->data[k][j];
				if (graph->data[i][j] > pathSum) {
					graph->data[i][j] = pathSum;
				}
			}
		}
	}
}

int main(int argc, char *argv[])
{
	int numVertices = 0;
	int showResults = 0;

	MPI_Init(&argc, &argv);

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
		std::cerr << "Usage: " << argv[0]
			  << "  [--show-results] <num_vertices>" << std::endl;
		MPI_Finalize();
		return 1;
	}

	std::cerr << "Running the Floyd-Warshall algorithm for a graph with "
		  << numVertices << " vertices." << std::endl;

	auto graph = createAndDistributeGraph(numVertices, 1 /* numProcesses */,
					      0 /* myRank */);
	if (graph == nullptr) {
		std::cerr << "Error distributing the graph for the algorithm."
			  << std::endl;
		MPI_Finalize();
		return 2;
	}

	if (showResults) {
		collectAndPrintGraph(graph, 1 /* numProcesses */,
				     0 /* myRank */);
	}

	double startTime = MPI_Wtime();

	runFloydWarshallSequential(graph);

	double endTime = MPI_Wtime();

	std::cerr << "The time required for the Floyd-Warshall algorithm on a "
		  << numVertices << "-node graph with " << 1
		  << " process(es): " << endTime - startTime << std::endl;

	if (showResults) {
		collectAndPrintGraph(graph, 1 /* numProcesses */,
				     0 /* myRank */);
	}

	destroyGraph(graph, 1 /* numProcesses */, 0 /* myRank */);

	MPI_Finalize();

	return 0;
}
