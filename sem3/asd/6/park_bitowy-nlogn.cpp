// Rozwiązanie O(nlogn)
// Dla każdego wierzchołka najdalszy od niego jest jeden z końców średnicy
// Średnica to ścieżka (diam[0], diam[1])
// Jeśli istnieje wierzchołek u taki, że odl(v, u) = d, to leży na dłuższej
// ze ścieżek (v, diam[0]), (v, diam[1])
// Wierzchoek u można znaleźć prostymi skokami binarnymi

#include <bits/stdc++.h>
using namespace std;

constexpr int MAX_N = 500007;
constexpr int LOG_N = 20;

struct Node {
  vector<int> adj;
  int pre, pos; // pre i post order
  int hop[LOG_N];
  int depth;
  int f, flen; // najdalszy wierzchołek i odległość do niego
};

vector<Node> G;

int nr, diam[2], diamlen = 0;

void dfs0(int v, int p) {
  G[v].pre = ++nr;
  G[v].hop[0] = p;
  for (int i = 1; i < LOG_N; i++)
    G[v].hop[i] = G[G[v].hop[i - 1]].hop[i - 1];

  G[v].depth = G[p].depth + 1;

  if (G[v].depth > diamlen)
    diamlen = G[v].depth - 1, diam[0] = v;

  for (auto u : G[v].adj)
    if (u != p)
      dfs0(u, v);

  G[v].pos = nr;
}

void dfs1(int v, int p, int num, int d = 0) {
  if (d > G[v].flen) {
    G[v].f = diam[num];
    G[v].flen = d;
  }

  if (num == 0 && d >= diamlen)
    diamlen = d, diam[1] = v;

  for (auto u : G[v].adj)
    if (u != p)
      dfs1(u, v, num, d + 1);
}

int lca(int v, int u) {
  if (G[v].pre <= G[u].pre && G[v].pos >= G[u].pos) return v;
  if (G[v].pre >= G[u].pre && G[v].pos <= G[u].pos) return u;
  for (int i = LOG_N - 1; i >= 0; i--)
    if (int w = G[v].hop[i]; G[w].pre > G[u].pre || G[w].pos < G[u].pos)
      v = w;
  return G[v].hop[0];
}

int go_up(int v, int d) {
  for (int i = LOG_N - 1; i >= 0; i--) {
    if (int w = G[v].hop[i]; G[v].depth - G[w].depth <= d) {
      d -= (G[v].depth - G[w].depth);
      v = w;
    }
  }
  return v;
}

int find_node(int v, int dist) {
  int u = G[v].f, w = lca(v, u);
  return dist <= G[v].depth - G[w].depth ? go_up(v, dist)
                                         : go_up(u, G[v].flen - dist);
}

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0), cout.tie(0);
  int n, m, a, b;
  cin >> n;
  G.resize(n + 7);

  for (int i = 1; i <= n; i++) {
    cin >> a >> b;
    if (a != -1) { G[i].adj.emplace_back(a), G[a].adj.emplace_back(i); }
    if (b != -1) { G[i].adj.emplace_back(b), G[b].adj.emplace_back(i); }
    G[i].flen = -1;
  }

  dfs0(1, 1);
  dfs1(diam[0], diam[0], 0);
  dfs1(diam[1], diam[1], 1);

  for (cin >> m; m--; ) {
    cin >> a >> b;
    cout << (G[a].flen < b ? -1 : find_node(a, b)) << '\n';
  }
}
