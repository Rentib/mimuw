#include <bits/stdc++.h>
using namespace std;

struct Node {
  vector<int> adj;
  vector<pair<int, int>> query; // number of query, distance
};

vector<int> ans, path;
vector<Node> G;
int diam[2], diamlen = -1;

void find_diameter(int v, int p, int depth, int num) {
  if (depth >= diamlen)
    diam[num] = v, diamlen = depth;
  for (auto u : G[v].adj)
    if (u != p)
      find_diameter(u, v, depth + 1, num);
}

void dfs(int v, int p) {
  path.emplace_back(v);
  for (auto [num, dist] : G[v].query)
    if (path.size() > dist)
      ans[num] = path[path.size() - dist - 1];
  for (auto u : G[v].adj)
    if (u != p)
      dfs(u, v);
  path.pop_back();
}

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0), cout.tie(0);
  int n, m;
  cin >> n;

  G.resize(n + 7);
  for (int i = 1, a, b; i <= n; i++) {
    cin >> a >> b;
    if (a != -1) { G[i].adj.emplace_back(a), G[a].adj.emplace_back(i); }
    if (b != -1) { G[i].adj.emplace_back(b), G[b].adj.emplace_back(i); }
  }
  cin >> m;
  for (int i = 0, v, dist; i < m; ans.emplace_back(-1)) {
    cin >> v >> dist;
    G[v].query.emplace_back(i, dist);
    i++;
  }

  find_diameter(1, 0, 0, 0);
  find_diameter(diam[0], 0, 0, 1);

  dfs(diam[0], 0);
  dfs(diam[1], 0);
  
  for (auto i : ans)
    cout << i << '\n';
}
