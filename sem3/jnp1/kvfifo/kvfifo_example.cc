#include "kvfifo.h"
#include "kwasow.h"
#include "tt.h"
#include <cassert>
#include <memory>
#include <vector>

auto f(kvfifo<int, int> q) { return q; }

int main() {
  kwasow::kwasowMain();
  ttt::tt_main();
  int keys[] = {3, 1, 2};

  kvfifo<int, int> kvf1 = f({});

  for (int i = 0; i < 3; ++i)
    kvf1.push(keys[i], i);

  auto &ref = kvf1.front().second;

  kvfifo<int, int> kvf2(kvf1); // Wykonuje się pełna kopia, dlaczego?
  kvfifo<int, int> kvf3;
  kvf3 = kvf2;

  ref = 10;
  assert(kvf1.front().second == 10);
  assert(kvf2.front().second != 10);

  kvf2.pop(); // Obiekt kvf2 dokonuje kopii i przestaje współdzielić dane z
              // kvf3.
  assert(kvf2.size() == 2);
  assert(kvf2.count(3) == 0);
  assert(kvf2.count(2) == 1);

  assert(kvf3.size() == 3);
  assert(kvf3.count(3) == 1);

  kvf2.push(1, 3);
  kvf2.move_to_back(1);
  assert(kvf2.size() == 3);
  assert(kvf2.front().second == 2 && kvf2.first(1).second == 1 &&
         kvf2.last(1).second == 3 && kvf2.back().second == 3);

  kvfifo<int, int> const kvf4 = kvf2;
  assert(kvf4.front().second == 2 && kvf4.first(1).second == 1 &&
         kvf4.last(1).second == 3 && kvf4.back().second == 3);

  int i = 1;
  for (auto k_it = kvf1.k_begin(), k_end = kvf1.k_end(); k_it != k_end;
       ++k_it, ++i)
    assert(i <= 3 && *k_it == i);

  auto kvf5 = std::make_unique<kvfifo<int, int>>();
  kvf5->push(4, 0);
  assert(kvf5->front().first == 4 && kvf5->front().second == 0);
  auto kvf6(*kvf5);
  kvf5.reset();
  assert(kvf6.front().first == 4 && kvf6.front().second == 0);

  std::swap(kvf1, kvf2);
  std::vector<kvfifo<int, int>> vec;
  for (int i = 0; i < 100000; i++)
    kvf1.push(i, i);
  for (int i = 0; i < 1000000; i++)
    vec.push_back(kvf1); // Wszystkie obiekty w vec współdzielą dane.
}
