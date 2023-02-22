#include <iostream>

using namespace std;
using ll = long long;

int main() {
  int n;
  cin >> n;

  int p = 0, k = n;

  while (p < k) {
    ll m = (p + k) >> 1;
    if (m * m <= n)
      p = m + 1;
    else
      k = m;
  }

  /* alternatywne rozwiązanie w czasie O(logn) gdzie n <= 1e9 */
  int res = 0;

  for (int i = 32; i >= 0; i--) {
    ll k = 1ll << i;
    if ((res + k) * (res + k) > n)
      continue;
    res += k;
  }

  cout << --p << ' ' << res << '\n';
}
