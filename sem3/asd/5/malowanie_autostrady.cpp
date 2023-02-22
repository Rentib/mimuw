#include <iostream>
using namespace std;
int M, tree[1 << 21], lazy[1 << 21];

void propagate(int v) {
  if (!lazy[v]) return;

  if (lazy[v] == 1) { // ustawiamy na 0
    lazy[v << 1 | 0] = lazy[v << 1 | 1] = 1;
    tree[v << 1 | 0] = tree[v << 1 | 1] = 0;
  }
  else { // ustawiamy na 1
    lazy[v << 1 | 0] = lazy[v << 1 | 1] = 2;
    tree[v << 1 | 0] = tree[v << 1 | 1] = tree[v] >> 1;
  }

  lazy[v] = 0;
}

void set(int a, int b, int c, int v = 1, int p = 1, int k = M) {
  if (p > b || k < a) return;
  if (a <= p && k <= b) {
    tree[v] = (k - p + 1) * c;
    lazy[v] = c + 1;
    return;
  }
  propagate(v);
  const int mid = (p + k) >> 1;
  set(a, b, c, v << 1 | 0, p, mid);
  set(a, b, c, v << 1 | 1, mid + 1,  k);
  tree[v] = tree[v << 1 | 0] + tree[v << 1 | 1];
}

int main() {
  int n, m;
  char c;
  cin >> n >> m;
  for (M = 1; M < n; M <<= 1);
  for (int i = 0, a, b; i < m; i++) {
    cin >> a >> b >> c;
    if (c == 'B') set(a, b, 1);
    if (c == 'C') set(a, b, 0);
    cout << tree[1] << '\n';
  }
}
