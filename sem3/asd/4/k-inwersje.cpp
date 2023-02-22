#include <iostream>

using namespace std;
using ll = long long;

constexpr int MAX_N = 20007, MOD = 1e9;
int n, k, a[MAX_N], M;
ll dp[2][MAX_N], tree[1 << 16], res;

void add(int d, int v, ll x) {
  for (v += M - 1; v; v >>= 1)
    tree[v] = (tree[v] + x) % MOD;
}

ll sum(int d, int p, int k, ll res = 0) {
  for (p += M - 1, k += M - 1; p <= k && p && k; p >>= 1, k >>= 1) {
    if ( (p & 1)) res = (res + tree[p++]) % MOD;
    if (!(k & 1)) res = (res + tree[k--]) % MOD;
  }
  return res;
}

int main() {
  cin >> n >> k;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    dp[1][i] = 1;
  }

  for (M = 1; M < n; M <<= 1);

  for (int d = 1; d < k; d++) {
    for (int i = 0; i < d; i++)
      add(d, a[i], dp[d & 1][i]);
    for (int i = d; i < n; i++) {
      dp[~d & 1][i] = sum(d, a[i] + 1, n);
      add(d, a[i], dp[d & 1][i]);
    }
    fill_n(dp[d & 1], n, 0);
    fill_n(tree, M << 1, 0);
  }

  for (int i = 0; i < n; i++)
    res = (res + dp[k & 1][i]) % MOD;

  cout << res << '\n';
}
