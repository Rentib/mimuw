#include <bits/stdc++.h>
using namespace std;
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0), cout.tie(0);
  int n, m, k;
  cin >> n >> m >> k;
  vector<int> G[n], p(n);
  for (int i = 0; i < n; i++)
    cin >> p[i];
  for (int i = 0, a, b; i < m; i++) {
    cin >> a >> b;
    G[--a].emplace_back(--b);
  }
  function<int(int)> maks = [&](int v) {
    for (auto u : G[v])
      p[v] = max(p[v], maks(u));
    G[v].clear();
    return p[v];
  };
  for (int i = 0; i < n; i++)
    p[i] = maks(i);
  sort(p.begin(), p.end());
  cout << p[--k] << '\n';
}
