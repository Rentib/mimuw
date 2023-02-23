#ifndef FUNCTIONAL_H_
#define FUNCTIONAL_H_

#include <functional>

inline auto compose() { return std::identity(); }

template <typename F>
auto compose(F f) {
  return std::bind(f, std::placeholders::_1);
}

template <typename F, typename... Args>
auto compose(F f, Args... args) {
  return std::bind(compose(args...), std::bind(f, std::placeholders::_1));
}

template <typename H, typename... Fs>
auto lift(H h, Fs... fs) {
  return [h, fs...](auto x) { return h(fs(x)...); };
}

#endif  // FUNCTIONAL_H_
