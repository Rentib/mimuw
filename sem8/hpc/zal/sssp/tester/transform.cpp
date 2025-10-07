#include <algorithm>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

using namespace std;
using ll = long long;
using ull = unsigned long long;

constexpr ll inf = 2e18;

static mt19937_64 gen;

ll random(ll l, ll r)
{
    uniform_int_distribution<ll> dis(l, r);
    return dis(gen);
}

int get_first(int numVertices, int numProcesses, int myRank)
{
    int base = numVertices / numProcesses;
    int remainder = numVertices % numProcesses;
    return myRank * base + std::min(myRank, remainder);
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <seed> <scale>" << endl;
        exit(1);
    }

    string filename = argv[1];
    int seed = 0;
    gen.seed(seed);

    // rmatX-<scale>.in
    int scale = 0;
    if (filename.find("rmat") != string::npos) {
        size_t pos = filename.find('-');
        if (pos != string::npos) {
            string scaleStr = filename.substr(pos + 1);
            cerr << "Scale: " << scaleStr << endl;
            scale = stoi(scaleStr);
        }
    }

    string type = filename.substr(0, filename.find('-'));

    ll n = 1 << scale;
    ll m = n * 16;
    vector<vector<pair<int, ll>>> G(n);
    vector<ll> dist(n, inf);

    // read from filename
    ifstream infile(filename);
    for (int i = 0; i < m; i++) {
        int u, v;
        infile >> u >> v;
        int w = random(0, 255);
        G[u].emplace_back(v, w);
        G[v].emplace_back(u, w);
    }

    priority_queue<pair<ll, int>, vector<pair<ll, int>>, greater<>> pq;
    dist[0] = 0;
    pq.push({0, 0});
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u])
            continue;

        for (const auto &[v, w] : G[u]) {
            if (dist[v] > d + w) {
                dist[v] = d + w;
                pq.push({dist[v], v});
            }
        }
    }

    vector<ll> ps;
    for (int i = 1; i <= 4; i++)
        ps.push_back(24 * i);
    for (ll p : ps) {
        string dirname = "./tests/" + type + "-" + to_string(seed) + "_" +
                         to_string(n) + "_" + to_string(p) + "/";
        system(("mkdir -p " + dirname).c_str());
        for (int rank = 0; rank < p; rank++) {
            ll first = get_first(n, p, rank);
            ll last = get_first(n, p, rank + 1) - 1;

            string filename = dirname + to_string(rank);
            ofstream in(filename + ".in");
            ofstream out(filename + ".out");

            in << n << " " << first << " " << last << endl;
            for (int u = first; u <= last; u++) {
                for (const auto &[v, w] : G[u])
                    in << u << " " << v << " " << w << endl;
                out << dist[u] << endl;
            }
        }
    }

    return 0;
}
