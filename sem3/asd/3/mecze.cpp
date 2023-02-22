/*
 * Można trzymać drużyny zawoników jako liczby binarne (m <= 50, więc wystarczy
 * uint64_t)
 * Jeśli zawodnik jest w lewej drużynie w k-tym meczu to zapalamy mu k-ty bit
 * Jeśli liczby dla dwóch zawodników są takie same, to grali oni zawsze w tych
 * samych drużynach i odpowiedź = NIE, a jeśli nie, to odpowiedź = TAK.
*/

#include <bits/stdc++.h>
using namespace std;

uint64_t teams[40007];

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0), cout.tie(0);
  int n, m, res = 0;

  for (cin >> n >> m; m--; )
    for (int i = 0, k; i < n; i++)
      if (cin >> k; i < n / 2)
        teams[--k] |= (1ULL << m);

  unordered_map<uint64_t, int> dinks;
  for (int i = 0; i < n; i++)
    res = max(res, dinks[teams[i]]++);

  cout << (!res ? "TAK\n" : "NIE\n");
}
