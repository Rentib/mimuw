#include <bits/stdc++.h>
using namespace std;
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0), cout.tie(0);
  int res = 1000007, i = 0, idx = 0;
  for (char c, p = '*'; cin >> c && c != EOF; i++) {
    if (c != '*') {
      if (p != '*' && p != c)
        res = min(res, i - idx);
      tie(p, idx) = {c, i};
    }
  }
  cout << i - min(res, i) + 1;
}
