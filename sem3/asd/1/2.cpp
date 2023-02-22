#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;
using ll = long long;

// Fibonacci: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...

ll podpunkt_b(int n) {
  static unordered_map<int, ll> m;
  if (n < 2)
    return n;
  else if (m[n])
    return m[n];

  ll k = (n + 1) >> 1;
  ll a = podpunkt_b(k), b = podpunkt_b(k - 1);

  return m[n] = (n & 1 ? (a * a + b * b) : (a * a + 2 * a * b));
}

struct Matrix {
  ll a, b, c, d;
  Matrix(){}
  Matrix(ll _a, ll _b, ll _c, ll _d): a(_a), b(_b), c(_c), d(_d){}
  Matrix operator *(const Matrix &m) const {
    return Matrix(a * m.a + b * m.c, a * m.b + b * m.d,
                  c * m.a + d * m.c, c * m.b + d * m.d);
  }
};

ll podpunkt_c(int n) {
  if (n < 2) return n;
  Matrix res = Matrix(1, 0, 1, 0);
  Matrix x   = Matrix(1, 1, 1, 0);

  for (--n; n; n >>= 1, x = x * x)
    if (n & 1)
      res = res * x;
  
  return res.a;
}

int main() {
  int n;
  cin >> n;
  cout << podpunkt_b(n) << ' ' << podpunkt_c(n) << '\n';
}
