#include <iostream>
#include <vector>

using namespace std;
using ll = long long;

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0), cout.tie(0);

  int n, m;
  cin >> n;

  vector<ll> a(n), res(n + 1, 0ll);
  
  for (int i = 0; i < n; i++)
    cin >> a[i];

  // filling candidates
  for (int k = 1; k <= n; k++)
    res[k] = res[k - 1] + a[n - k];

  int su[2] = { n, n }; // index of smallest   used element of [parity]
  int gu[2] = { n, n }; // index of greatest unused element of [parity]

  while (--gu[0] >= 0 && (a[gu[0]] & 1) != 0);
  while (--gu[1] >= 0 && (a[gu[1]] & 1) != 1);

  res[0] = -1;
  for (int beg = n - 1, k = 1; beg >= 0; beg--, k++) {
    int p = a[beg] & 1; // parity of new element
    su[p] = beg;
    while (--gu[p] >= 0 && (a[gu[p]] & 1) != p);

    if (res[k] & 1)
      continue;

    ll c01 = -1; // change 0 to 1
    ll c10 = -1; // change 1 to 0
    if (su[0] < n && gu[1] > -1)
      c01 = res[k] - a[su[0]] + a[gu[1]];
    if (su[1] < n && gu[0] > -1)
      c10 = res[k] - a[su[1]] + a[gu[0]];

    res[k] = max(c01, c10);
  }

  cin >> m;
  for (int i = 0, q; i < m; i++) {
    cin >> q;
    cout << res[q] << '\n';
  }

  return 0;
}
