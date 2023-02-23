#ifndef TT_H
#define TT_H

#include "kvfifo.h"
#include <cassert>
#include <map>

namespace ttt {
bool b = false;
class mv {
public:
  mv() = default;
  mv([[maybe_unused]] mv &&m) {
    if (b) {
      throw std::runtime_error{"move"};
    }
  }
  mv([[maybe_unused]] const mv &m) {
    if (b) {
      throw std::runtime_error{"copy"};
    }
  }

  mv &operator=([[maybe_unused]] mv &&m) {
    if (b) {
      throw std::runtime_error{"move assignment"};
    }
  }
  mv &operator=([[maybe_unused]] const mv &m) {
    if (b) {
      throw std::runtime_error{"copy assignment"};
    }
  }
};

void tt_main() {
  // kvfifo<int, mv> q{};
  // for (size_t i = 0; i < 10; i++) {
  //   q.push(i, {});
  // }

  // b = true;
  // q.move_to_back(5);

  // kvfifo<int, int> q({});
  // for (int i = 0; i < 10; i++)
  //   q.push(i & 1, i);

  // q.move_to_back(0);

  // for (int i = 0; i < 10; i++) {
  //   auto p = q.front();
  //   q.pop();
  //   printf("%d ", p.second);
  // }

  std::map<int, int> m0{};
  m0.insert({{1, 1}, {2, 2}, {3, 3}, {4, 4}});

  std::map<int, int> m1{m0};
  m1.insert({{6, 6}});

  assert(m1.begin() != m0.begin());
}

} // namespace ttt

#endif