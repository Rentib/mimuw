#include <bits/stdc++.h>
using namespace std;
using ll = long long;
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0), cout.tie(0);
  int n, d, v;
  cin >> n;
  vector<pair<ll, ll>> p(n), sx(n), sy(n);
  for (int i = 0; i < n; i++) {
    cin >> p[i].first >> p[i].second;
    sx[i] = make_pair(p[i].first,  i);
    sy[i] = make_pair(p[i].second, i);
  }
  sort(sx.begin(), sx.end());
  sort(sy.begin(), sy.end());
  vector<pair<int, ll>> G[n];
  auto add = [&](int v, int u) {
    int w = min(abs(p[v].first - p[u].first), abs(p[v].second - p[u].second));
    G[v].emplace_back(u, w);
    G[u].emplace_back(v, w);
  };
  for (int i = 1; i < n; i++) {
    add(sx[i - 1].second, sx[i].second);
    add(sy[i - 1].second, sy[i].second);
  }
  vector<ll> dist(n, 1e9);
  priority_queue<pair<ll, int>> q;
  q.emplace(0, 0);
  dist[0] = 0;
  while (!q.empty()) {
    tie(d, v) = q.top();
    q.pop();
    for (auto [u, w] : G[v]) {
      if (w - d < dist[u]) {
        dist[u] = w - d;
        q.emplace(d - w, u);
      }
    }
  }
  cout << dist[n - 1] << '\n';
}
