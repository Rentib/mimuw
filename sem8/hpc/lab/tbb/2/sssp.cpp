#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_priority_queue.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>

using namespace std;

const int INF = numeric_limits<int>::max();
int n = 1000;                            // Number of nodes
int m = 5000;                            // Number of edges
vector<vector<pair<int, int>>> graph(n); // Adjacency list

vector<int> par_dijkstra()
{
    tbb::concurrent_unordered_map<int, int> dist;
    for (int i = 0; i < n; i++)
        dist[i] = INF;
    dist[0] = 0;

    using pii = pair<int, int>; // {distance, node}
    struct Compare {
        bool operator()(const pii &a, const pii &b) const
        {
            return a.first > b.first; // min-heap
        }
    };

    tbb::concurrent_priority_queue<pii, Compare> pq;
    pq.push({0, 0});

    while (!pq.empty()) {
        pii current;
        if (!pq.try_pop(current))
            continue;

        int d = current.first;
        int u = current.second;

        if (d > dist[u])
            continue;

        tbb::parallel_for_each(graph[u].begin(), graph[u].end(),
                               [&](const pair<int, int> &edge) {
                                   int v = edge.first;
                                   int weight = edge.second;
                                   int new_dist = d + weight;
                                   int old_dist = dist[v];
                                   while (new_dist < old_dist) {
                                       if (dist[v] == old_dist) {
                                           dist[v] = new_dist;
                                           pq.push({new_dist, v});
                                           break;
                                       } else {
                                           old_dist = dist[v];
                                       }
                                   }
                               });
    }

    vector<int> result(n, INF);
    for (int i = 0; i < n; ++i)
        result[i] = dist[i];
    return result;
}

vector<int> seq_dijkstra()
{
    vector<int> dist(n, INF);
    dist[0] = 0; // source node
    priority_queue<pair<int, int>, vector<pair<int, int>>,
                   greater<pair<int, int>>>
        pq;
    pq.push({0, 0}); // {distance, node}
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u])
            continue; // Skip outdated entries

        for (auto [v, weight] : graph[u]) {
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

int main()
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, n - 1);
    for (int i = 0; i < m; ++i) {
        int u = dis(gen);
        int v = dis(gen);
        int weight = dis(gen) + 1;
        graph[u].emplace_back(v, weight);
        graph[v].emplace_back(u, weight); // Undirected
    }

    auto par_dist = par_dijkstra(); // Parallel Dijkstra
    auto seq_dist = seq_dijkstra(); // Sequential Dijkstra

    // Output
    cout << "Distances from source node 0:\n";
    cout << "Vertex, Parallel, Sequential\n";
    for (int i = 0; i < min(n, 10); ++i)
        printf("%6d, %8d, %10d\n", i, par_dist[i], seq_dist[i]);
    for (int i = 0; i < n; i++) {
        if (par_dist[i] != seq_dist[i]) {
            cout << "Mismatch at vertex " << i << ": "
                 << "Bellman-Ford = " << par_dist[i]
                 << ", Dijkstra = " << seq_dist[i] << endl;
        }
    }

    return 0;
}
