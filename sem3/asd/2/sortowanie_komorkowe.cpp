#include <iostream>
using namespace std;
using ll = long long;

constexpr int MAX_N = 1007, MOD = 1e9;

#define ADD(a, b) do { a = (a + b) % MOD; } while (false);

ll n, t[MAX_N], dp[2][2][MAX_N]; // [#elements & 1][side][index]

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0), cout.tie(0);
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> t[i];
    dp[1][0][i] = 0;
    dp[1][1][i] = 1;
  }

  for (int k = 1; k++ < n; ) {
    for (int i = 0; i < n; i++)
      dp[k & 1][0][i] = dp[k & 1][1][i] = 0;
    for (int l = 0, r = k - 1; r < n; l++, r++) {
      if (t[l] < t[l + 1])
        ADD(dp[k & 1][0][l], dp[~k & 1][0][l + 1]);
      if (t[l] < t[r])
        ADD(dp[k & 1][0][l], dp[~k & 1][1][r]);

      if (t[r] > t[r - 1])
        ADD(dp[k & 1][1][r], dp[~k & 1][1][r - 1]);
      if (t[r] > t[l])
        ADD(dp[k & 1][1][r], dp[~k & 1][0][l]);
    }
  }

  cout << (dp[n & 1][0][0] + dp[n & 1][1][n - 1]) % MOD;
}
