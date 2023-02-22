#include <bits/stdc++.h>
using namespace std;
constexpr int MAX_N = 1000007;
vector<int> s[MAX_N];
int n, cnt, F[MAX_N];
int Find(int a) {
  return F[a] == a ? a : F[a] = Find(F[a]);
}
void Union(int a, int b) {
  a = Find(a);
  b = Find(b);
  F[max(a, b)] = min(a, b);
}
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0), cout.tie(0);
  cin >> n;
  iota(F, F + n + 1, 0);
  priority_queue<int> q;
  q.emplace(0);
  for (int i = 0, a; i < n; i++) {
    cin >> a;
    if (int f = q.top(); f > a) {
      while (q.top() > a) {
        Union(q.top(), a);
        q.pop();
      }
      a = f;
    }
    q.emplace(a);
  }
  for (int i = 1; i <= n; i++) {
    int f = Find(i);
    cnt += (i == f);
    s[f - 1].emplace_back(i);
  }
  cout << cnt << '\n';
  for (int i = 0; i < n; i++) {
    if (!s[i].empty()) {
      cout << s[i].size();
      for (auto j : s[i])
        cout << ' ' << j;
      cout << '\n';
    }
  }
}
