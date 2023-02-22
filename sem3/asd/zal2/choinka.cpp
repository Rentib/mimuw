#include <iostream>
#include <vector>

using namespace std;

constexpr int MAX_N = 200007;

vector<int> G[MAX_N];
int pre[MAX_N], pos[MAX_N], nr;

void dfs(int v, int o) {
  pre[v] = ++nr;
  for (auto u : G[v]) {
    if (u != o)
      dfs(u, v);
  }
  pos[v] = nr;
}

struct node {
  int max, max_cnt;
  int min, min_cnt;

  node() : max(0), max_cnt(0), min(1000000001), min_cnt(0) {}
  node(int x) : max(x), max_cnt(1), min(x), min_cnt(1) {}
};

node merge(const node &a, const node &b) {
  node v;
  v.max = max(a.max, b.max);
  v.min = min(a.min, b.min);
  v.max_cnt = (a.max == v.max ? a.max_cnt : 0)
            + (b.max == v.max ? b.max_cnt : 0);
  v.min_cnt = (a.min == v.min ? a.min_cnt : 0)
            + (b.min == v.min ? b.min_cnt : 0);
  return v;
}

node tree[1 << 20];
int M = 1;

void update(int v, int c) {
  tree[v += M - 1] = node(c);
  while (v >>= 1)
    tree[v] = merge(tree[v << 1 | 0], tree[v << 1 | 1]);
}

node query(int p, int k) {
  node res;
  for (p += M - 1, k += M - 1; p <= k; p >>= 1, k >>= 1) {
    if ((p & 1))
      res = merge(res, tree[p++]);
    if (!(k & 1))
      res = merge(res, tree[k--]);
  }
  return res;
}

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0), cout.tie(0);
  int n, q;
  cin >> n >> q;
  for (int i = 2, o; i <= n; i++) {
    cin >> o;
    G[o].emplace_back(i);
    G[i].emplace_back(o);
  }

  dfs(1, 0);

  while (M < n)
    M <<= 1;

  for (int i = 1, x; i <= n; i++) {
    cin >> x;
    update(pre[i], x);
  }

  char t;
  for (int i = 0, v, x; i < q; i++) {
    cin >> t;
    if (t == 'z') {
      cin >> v >> x;
      update(pre[v], x);
    } else {
      cin >> v;
      node res = query(pre[v], pos[v]);
      cout << ((res.min == res.max ||
                (res.max_cnt + res.min_cnt == pos[v] - pre[v] + 1 &&
                 min(res.min_cnt, res.max_cnt) == 1))
                   ? "TAK\n"
                   : "NIE\n");
    }
  }
}
