#include <iostream>
using namespace std;
int main() {
  // Zwykły algorytm Kadane (maksymalna suma spójnego podciągu)
  int n;
  cin >> n;

  int res = 0, sum = 0;
  for (int i = 0, k; i < n; i++) {
    cin >> k;
    sum = max(0, sum + k);
    res = max(res, sum);
  }

  cout << res;
}
