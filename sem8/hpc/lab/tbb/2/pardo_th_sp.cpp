#include "tbb/tbb.h"
#include <iostream>
#include <math.h>
#include <map>
#include <set>
#include <random> 

typedef int node_t;


// Our graph just maps a node onto its neighbors.
typedef std::map<node_t, std::set<node_t>> graph_t;


// Initialize a DAG with random adjacencies.
void rand_init_DAG_graph(graph_t& graph, int node_count,
                         double edge_probability) {
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<> dis{0, 1};
    for (int i = 0; i < node_count; ++i) {
        auto neighbors = std::set<node_t>();
        for (int j = i+1; j < node_count; ++j) {
            if (dis(gen) < edge_probability) {
                neighbors.insert(j);
            }
        }
        graph[i] = neighbors;
    }
}


// Traverse the DAG - may visit a node multiple times.
void seq_traverse(const node_t& node, graph_t& graph, int& edge_count) {
    std::cout << "Now processing: " << node << "." << std::endl;
    for (const auto& neighbor : graph[node]) {
        std::cout << "edge: " << node << "->" << neighbor << std::endl;
        edge_count++;
        seq_traverse(neighbor, graph, edge_count);
    }
}


int main(int argc, char* argv[]) {
    const int node_count = 10;
    graph_t graph;
    rand_init_DAG_graph(graph, node_count, 0.5);
    node_t node = 0;
    int seq_edge_count = 0;
    seq_traverse(node, graph, seq_edge_count);

    std::mutex stdout_mutex;
    tbb::enumerable_thread_specific<int> counters;
    tbb::parallel_for_each(&node, &node+1, [&graph, &counters, &stdout_mutex]
                           (const node_t& node, tbb::feeder<node_t>& feeder) {
            {
                // this mutex is just for stdout formatting
                std::scoped_lock lock(stdout_mutex);
                std::cout << "Now processing: " << node << "." << std::endl;
            }  // scoped_lock mutex will be released here
            for (const auto& neighbor : graph[node]) {
                {
                    // this mutex is just for stdout formatting
                    std::scoped_lock lock(stdout_mutex);
                    std::cout << "edge: " << node << "->" << neighbor
                              << std::endl;
                }  // scoped_lock mutex will be released here
                (counters.local())++;
                feeder.add(neighbor);
            }
        }
        );
    int par_edge_count = 0;
    for (const auto& counter : counters) {
        par_edge_count += counter;
    }
    std::cout << "Edges traversed: sequentially: "<< seq_edge_count
              << " in prallel: " <<par_edge_count << std::endl;
}
