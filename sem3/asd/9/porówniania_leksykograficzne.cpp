#include <iostream>
#include <vector>
using namespace std;
using ll = long long;
constexpr int MAX_N = 300007, q = 1000000007;
int main() {
  int n, m;
  string s;
  cin >> n >> m >> s;
  vector<ll> h(n + 1, 0), p(n + 1, 1);
  #define HASH(a, b, c)    (((h[b] - h[a - 1] + q) * p[c]) % q)
  for (ll i = 0; i < n; i++) {
    h[i + 1] = (h[i] + (s[i] - 'a') * p[i]) % q;
    p[i + 1] = (p[i] * 997) % q;
  }
  auto cmp = [&](int a, int b, int c, int d) -> int { // a <= c
    int low = 0, high = min(b - a, d - c), diff = c - a;
    if (b - a == d - c && HASH(a, b, diff) == HASH(c, d, 0)) return 0;
    while (low < high) {
      int mid = (low + high) >> 1;
      if (HASH(a, a + mid, diff) == HASH(c, c + mid, 0)) low = mid + 1;
      else high = mid;
    }
    return s[a - 1 + low] == s[c - 1 + low] ? b - a < d - c ? -1 : 1
         : s[a - 1 + low] <  s[c - 1 + low] ? -1 : 1;
  };
  for (int a, b, c, d; m--; ) {
    cin >> a >> b >> c >> d;
    int k = (a <= c ? cmp(a, b, c, d) : -cmp(c, d, a, b));
    switch (k) {
      case -1: cout << "<\n"; break;
      case  0: cout << "=\n"; break;
      case  1: cout << ">\n"; break;
    }
  }
}
