#include <bits/stdc++.h>
using namespace std;

int n;
string s;

bool check(int k) {
  int distinct = 0;
  vector<int> alphabet('Z' - 'A'+ 1, 0);

  for (int i = 0; i < k; i++)
    if (s[i] != '*') distinct += (++alphabet[s[i] - 'A'] == 1);

  for (int l = 0, r = k - 1; distinct < 2 && ++r < n; l++) {
    if (s[l] != '*') distinct -= (--alphabet[s[l] - 'A'] == 0);
    if (s[r] != '*') distinct += (++alphabet[s[r] - 'A'] == 1);
  }

  return distinct < 2;
}

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0), cout.tie(0);

  cin >> s;

  n = s.size();
  int low = 1, high = n;
  while (low < high) {
    int mid = low + ((high - low) >> 1);
    if (!check(n - mid + 1))
      low = mid + 1;
    else
      high = mid;
  }
  cout << low << '\n';
}
