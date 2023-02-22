#include <iostream>
#include <vector>

using namespace std;

int kth(vector<int> &v, int k) {
  int pivot = v[0];
  vector<int> w[3];
  for (auto i : v)
    w[i < pivot ? 0 : i > pivot ? 2 : 1].emplace_back(i);
  for (int i = 0; i < 2; k -= w[i++].size())
    if (k < w[i].size())
      return i == 1 ? pivot : kth(w[i], k);
  return kth(w[2], k);
}

int main() {
  int n, k;
  cin >> n >> k;
  vector<int> v(n);
  for (int i = 0; i < n; i++)
    cin >> v[i];
  
  cout << kth(v, k) << '\n';
}
