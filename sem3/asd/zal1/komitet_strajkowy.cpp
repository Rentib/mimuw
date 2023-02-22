#include <bits/stdc++.h>
using namespace std;

constexpr int MOD = 1'000'000'007, INF = 1 << 30;

pair<int, int> merge(pair<int, int> a, pair<int, int> b) {
  if (a.first == b.first) {
    int s = a.second + b.second;
    if (s >= MOD)
      s -= MOD;
    return make_pair(a.first, s);
  }
  return a.first < b.first ? a : b;
}

struct SegTree {
  vector<pair<int, int>> tree;
  int M;
  SegTree(int n) {
    for (M = 1; M < n; M <<= 1) {}
    tree.resize(M * 2, make_pair(INF, 0));
  }
  void insert(int v, pair<int, int> t) {
    v += M;
    tree[v] = t;
    for (v /= 2; v; v >>= 1) tree[v] = merge(tree[v * 2], tree[v * 2 + 1]);
  }
  pair<int, int> query(int l, int r) {
    l += M, r += M;
    pair<int, int> answer = tree[l];
    if (l != r)
      answer = merge(answer, tree[r]);
    while (l / 2 != r / 2) {
      if (l % 2 == 0)
        answer = merge(answer, tree[l + 1]);
      if (r % 2 == 1)
        answer = merge(answer, tree[r - 1]);
      l /= 2;
      r /= 2;
    }
    return answer;
  }
};

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0), cout.tie(0);
  int n, k, l;
  cin >> n >> k >> l;
  l--;
  // we decrease l by one in order to match our formulas
  vector<int> a(n);
  for (int &x : a)
    cin >> x;
  a.push_back(-INF);
  sort(a.begin(), a.end());
  vector<int> dp_moves(n + 1, INF);
  // dp_moves[i] -> how many moves do we need to cover a prefix
  // of employees such that the last employee in the committee is the
  // i-th employee
  vector<int> dp_ways(n + 1, 0);
  // dp_ways[i] -> in how many ways can we achieve the optimal number
  // of moves in dp_moves[i]

  dp_moves[0] = 0;
  // we select the 0-th employee employee (it's the one with rank of -inf)
  dp_ways[0] = 1;
  // there is only one way to select this employee

  SegTree T(n + 1);
  T.insert(0, make_pair(dp_moves[0], dp_ways[0]));

  for (int i = 1; i <= n; i++) {

    int rightmost = (lower_bound(a.begin(), a.end(), a[i] - l) - a.begin()) - 1;
    // if we take the i-th guy to the committee
    // what is the closest to the left that can
    // be in the committee as well

    int first_out_of_range = (lower_bound(a.begin(), a.end(), a[i] - k) - a.begin()) - 1;
    // the first guy to the left that is out of
    // the range of the-ith guy

    int leftmost = lower_bound(a.begin(), a.end(), a[first_out_of_range] - k) - a.begin();
    // to form a correct committee we need to choose
    // a guy that will cover the first_out_of_range
    // guy so it has to be a guy whose index is at
    // least leftmost

    if (leftmost <= rightmost) {
      // we take the least number of moves in a tree and sum the dp_ways over
      // these indices
      tie(dp_moves[i], dp_ways[i]) = T.query(leftmost, rightmost);
      T.insert(i, make_pair(++dp_moves[i], dp_ways[i]));
    }
  }

  pair<int, int> answer(INF, 0);
  for (int i = 1; i <= n; i++) {
    if (a.back() - a[i] <= k)
      answer = merge(answer, make_pair(dp_moves[i], dp_ways[i]));
  }
  // we choose the answer among the guys that can cover the rightmost guy

  cout << answer.first << ' ' << answer.second << '\n';

  return 0;
}
