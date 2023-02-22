#include <iostream>
#include <vector>
using namespace std;
int main() {
  // Rozwiązanie w oparciu o algorytm głosowania Moore'a do wyznaczania lidera.
  // Złożoność obliczeniowa: O(n);
  // Złożoność pamięciowa:   O(1);

  ios_base::sync_with_stdio(0);
  cin.tie(0), cout.tie(0);

  int n;
  cin >> n;
  vector<int> a(n);
  for (int i = 0; i < n; i++)
    cin >> a[i];

  if (n == 1) {
    cout << a[0];
    return 0;
  }

  int cnt1 = 0, cnt2 = 0;
  int k1 = a[0] - 1, k2 = a[0] + 1;

  for (auto i : a) {
    if (!cnt1 && i != k2)
      k1 = i;
    else if (!cnt2 && i != k1)
      k2 = i;

    if (i == k1)
      cnt1++;
    else if (i == k2)
      cnt2++;
    else {
      cnt1--;
      cnt2--;
    }
  }

  cnt1 = 0;
  cnt2 = 0;

  for (auto i : a) {
    cnt1 += (k1 == i);
    cnt2 += (k2 == i);
  }

  cout << (cnt1 > cnt2 ? k1 : k2) << '\n';
}  
